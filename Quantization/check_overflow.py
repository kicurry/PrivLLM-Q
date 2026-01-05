import os
import sys
import random
import numpy as np
import torch
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, init_k_quantizer, init_v_quantizer
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from main import evaluate
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention
from utils.quant_utils import QKRotationWrapper
import quantize.int_linear_fake as int_linear_fake
from utils.quant_utils import init_s_quantizer
from quantize.quantizer import UniformAffineQuantizer
import functools
from safetensors.torch import load_file
import ast
import scipy.stats as st
torch.backends.cudnn.benchmark = True

def compute_bacc(model):
    b_acc_map = {}
    for _,module in model.named_modules():
        if isinstance(module,LlamaDecoderLayer):
            for name,layer in module.named_modules():
                if isinstance(layer,int_linear_fake.QuantLinear):
                    weight_int = layer.weight_quantizer.get_int(layer.weight).to(torch.float64)
                    print("name,weight_int.shape",name,weight_int.shape)
                    weight_neg = torch.where(weight_int < 0, weight_int, torch.zeros_like(weight_int))
                    weight_pos = torch.where(weight_int > 0, weight_int, torch.zeros_like(weight_int))
                    weight_neg = torch.sum(torch.abs(weight_neg),dim=-1,keepdim=True)
                    weight_pos = torch.sum(weight_pos,dim=-1,keepdim=True)
                    weight_max = torch.max(weight_neg,weight_pos)
                    b_acc = int(4 + torch.max(torch.ceil(torch.log2(weight_max))))
                    if name in b_acc_map:
                        b_acc_map[name] = max(b_acc_map[name],b_acc)
                    else:
                        b_acc_map[name] = b_acc
    return b_acc_map

stat = {}
def add_overflow_hook(model):
    
    def overflow_stat(module, x, name):
        # print(name)
        x = x[0]
        # print(x.shape)
        x_int = module.get_int(x)
        # print(x_int.shape)
        ema_factor = 0.99
        # x_below_qmin = torch.where(x_int < module.qmin, x_int, torch.zeros_like(x_int))
        # x_above_qmax = torch.where(x_int > module.qmax, x_int, torch.zeros_like(x_int))
        if name+"_min" not in stat:
            stat[name+"_min"] = torch.min(x_int)
        else:
            stat[name+"_min"] = torch.min(stat[name+"_min"],torch.min(x_int))
        if name+"_max" not in stat:
            stat[name+"_max"] = torch.max(x_int)
        else:
            stat[name+"_max"] = torch.max(stat[name+"_max"],torch.max(x_int))
        if name+"_var" not in stat:
            stat[name+"_var"] = torch.var(x_int)
        else:
            stat[name+"_var"] = ema_factor * stat[name+"_var"] + (1-ema_factor) * torch.var(x_int)
        if name+"_mean" not in stat:
            stat[name+"_mean"] = torch.mean(x_int)
        else:
            stat[name+"_mean"] = ema_factor * stat[name+"_mean"] + (1-ema_factor) * torch.mean(x_int)
        
        # for bit in [4,5,6,7,8]:
        #     lower_bound = -2**(bit-1)
        #     upper_bound = 2**(bit-1) - 1
        #     x_in_range_rate = torch.sum((x_int >= lower_bound) & (x_int <= upper_bound))/torch.numel(x_int)
        #     x_out_range_rate = 1 - x_in_range_rate
        #     if name + "_out_"+str(bit) not in stat:
        #         stat[name + "_out_"+str(bit)] = x_out_range_rate
        #     else:
        #         stat[name + "_out_"+str(bit)] = ema_factor * stat[name + "_out_"+str(bit)] + (1-ema_factor) * x_out_range_rate
        #     # print(f"name:{name}, bit:{bit}, x_out_range_rate:{x_out_range_rate}, ema:{stat[name + '_out_'+str(bit)]}")
    for name,module in model.named_modules():
        if isinstance(module, UniformAffineQuantizer) and module.quant_type == "activation":
            print("name:",name)
            module.register_forward_pre_hook(functools.partial(overflow_stat, name=name))
    # exit(0)

def process_stat(model_name):
    if model_name == "llama3":
        stat = torch.load("llama3_stat.pth")
    elif model_name == "llama2":
        stat = torch.load("llama2_stat.pth")
    else:
        raise ValueError(f"Unsupported model: {name}")
    processed_stat = [{} for _ in range(32)]
    for name,value in stat.items():
        # print(name)
        layer_idx = int(name.split(".")[2])
        # print(layer_idx)
        if "input_layernorm.output_quantizer_var" in name:
            processed_stat[layer_idx]["qkv_var"] = value
        elif "v_proj.output_quantizer_var" in name:
            processed_stat[layer_idx]["v_var"] = value
        elif "k_quantizer_var" in name:
            processed_stat[layer_idx]["k_var"] = value
        elif "q_quantizer_var" in name:
            processed_stat[layer_idx]["q_var"] = value
        elif "s_quantizer_var" in name:
            processed_stat[layer_idx]["s_var"] = value
        elif "o_proj.input_quantizer_var" in name:
            processed_stat[layer_idx]["o_var"] = value
        elif "post_attention_layernorm.output_quantizer_var" in name:
            processed_stat[layer_idx]["up_gate_var"] = value
        elif "down_proj.input_quantizer_var" in name:
            processed_stat[layer_idx]["down_var"] = value
        
        if "input_layernorm.output_quantizer_mean" in name:
            processed_stat[layer_idx]["qkv_mean"] = value
        elif "v_proj.output_quantizer_mean" in name:
            processed_stat[layer_idx]["v_mean"] = value
        elif "k_quantizer_mean" in name:
            processed_stat[layer_idx]["k_mean"] = value
        elif "q_quantizer_mean" in name:
            processed_stat[layer_idx]["q_mean"] = value
        elif "s_quantizer_mean" in name:
            processed_stat[layer_idx]["s_mean"] = value
        elif "o_proj.input_quantizer_mean" in name:
            processed_stat[layer_idx]["o_mean"] = value
        elif "post_attention_layernorm.output_quantizer_mean" in name:
            processed_stat[layer_idx]["up_gate_mean"] = value
        elif "down_proj.input_quantizer_mean" in name:
            processed_stat[layer_idx]["down_mean"] = value
    if model_name == "llama3":
        torch.save(processed_stat, "llama3_processed_stat.pth")
    elif model_name == "llama2":
        torch.save(processed_stat, "llama2_processed_stat.pth")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    print(processed_stat)

def compute_bound(model, model_name):
    stat = torch.load(f"{model_name}_processed_stat.pth")
    res = [{} for _ in range(32)]
    for name,layer in model.named_modules():
        if isinstance(layer,int_linear_fake.QuantLinear):
            # print("name:",name)
            layer_idx = int(name.split(".")[2])
            weight_int = layer.weight_quantizer.get_int(layer.weight).to(torch.float64)
            # print(weight_int.shape)
            # Compute L1 and L2 norms along the input dimension (dim=1) for each output neuron
            l1_norms = torch.norm(weight_int, p=1, dim=1).detach().cpu().numpy()  # [out_features]
            l2_norms = torch.norm(weight_int, p=2, dim=1).detach().cpu().numpy()  # [out_features]
            # Take the maximum ratio across all output neurons (dim=0)
            tmp_k = np.min(l1_norms / l2_norms)
            # print(tmp_k)
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                sigma_x = torch.sqrt(stat[layer_idx]["qkv_var"]).detach().cpu().numpy()
                mean_x = stat[layer_idx]["qkv_mean"].detach().cpu().numpy()
            elif "o_proj" in name:
                sigma_x = torch.sqrt(stat[layer_idx]["o_var"]).detach().cpu().numpy()
                mean_x = stat[layer_idx]["o_mean"].detach().cpu().numpy()
            elif "up_proj" in name or "gate_proj" in name:
                sigma_x = torch.sqrt(stat[layer_idx]["up_gate_var"]).detach().cpu().numpy()
                mean_x = stat[layer_idx]["up_gate_mean"].detach().cpu().numpy()
            elif "down_proj" in name:
                sigma_x = torch.sqrt(stat[layer_idx]["down_var"]).detach().cpu().numpy()
                mean_x = stat[layer_idx]["down_mean"].detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported layer: {name}")
            k = 8/sigma_x * tmp_k
            weight_int = weight_int.detach().cpu().numpy()
            delta = mean_x * np.sum(weight_int, axis=1) / (sigma_x * l2_norms)
            delta = np.max(delta)
            delta = delta.astype(np.float64)
            k = k.astype(np.float64)
            p_overflow = 1-st.norm.cdf(k-delta) + st.norm.cdf(-k-delta)
            res[layer_idx][name] = p_overflow
            res[layer_idx][name+"_mean"] = mean_x
            res[layer_idx][name+"_sigma"] = sigma_x
        elif isinstance(layer, QKRotationWrapper):
            layer_idx = int(name.split(".")[2])
            sigma_q = torch.sqrt(stat[layer_idx]["q_var"]).detach().cpu().numpy()
            sigma_k = torch.sqrt(stat[layer_idx]["k_var"]).detach().cpu().numpy()
            mean_q = stat[layer_idx]["q_mean"].detach().cpu().numpy()
            mean_k = stat[layer_idx]["k_mean"].detach().cpu().numpy()
            mean_z = 128 * mean_q * mean_k
            sigma_z = np.sqrt(128 * (sigma_q**2 * sigma_k**2 + sigma_q**2 * mean_k**2 + sigma_k**2 * mean_q**2))
            sigma_z = np.max(sigma_z)
            sigma_q = sigma_q.astype(np.float32)
            sigma_k = sigma_k.astype(np.float32)
            T_bound = 128 * (2 ** 3 -1 )*(2 ** 3 -1 )
            p_overflow = 1-st.norm.cdf((T_bound-mean_z)/sigma_z) + st.norm.cdf((-T_bound-mean_z)/sigma_z)
            res[layer_idx][name] = p_overflow
            res[layer_idx][name+"_mean"] = mean_z
            res[layer_idx][name+"_sigma"] = sigma_z
        elif "s_quantizer" in name:            
            layer_idx = int(name.split(".")[2])
            sigma_v = torch.sqrt(stat[layer_idx]["v_var"]).detach().cpu().numpy()
            sigma_s = torch.sqrt(stat[layer_idx]["s_var"]).detach().cpu().numpy()
            mean_v = stat[layer_idx]["v_mean"].detach().cpu().numpy()
            mean_s = stat[layer_idx]["s_mean"].detach().cpu().numpy()
            mean_z = 128 * mean_v * mean_s
            sigma_z = np.sqrt(128 * (sigma_v**2 * sigma_s**2 + sigma_v**2 * mean_s**2 + sigma_s**2 * mean_v**2))
            sigma_z = np.max(sigma_z)
            sigma_v = sigma_v.astype(np.float32)
            sigma_s = sigma_s.astype(np.float32)
            T_bound = 128 * (2 ** 7 -1 )*(2 ** 3 -1 )
            p_overflow = 1-st.norm.cdf((T_bound-mean_z)/sigma_z) + st.norm.cdf((-T_bound-mean_z)/sigma_z)
            res[layer_idx][name] = p_overflow   
            res[layer_idx][name+"_mean"] = mean_z
            res[layer_idx][name+"_sigma"] = sigma_z
    print(res)
    torch.save(res, f"{model_name}_bound.pth")
            

def compute_worse_prob(model_name):
    if model_name == "llama3":
        stat = torch.load("llama3_bound.pth",weights_only=False)
    elif model_name == "llama2":
        stat = torch.load("llama2_bound.pth",weights_only=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Dictionary to collect values for each component across all layers
    component_k_values = {}  # For k values (smaller is better)
    component_p_values = {}  # For probability values (larger is worse)
    
    # Iterate through all 32 layers
    for layer_idx in range(32):
        layer_data = stat[layer_idx]
        
        for key, value in layer_data.items():
            # Extract component name by removing the layer-specific prefix
            # e.g., "model.layers.0.self_attn.q_proj" -> "self_attn.q_proj"
            component_name = ".".join(key.split(".")[3:])  # Skip "model.layers.X"
            
            # This is a probability value
            if component_name not in component_p_values:
                component_p_values[component_name] = []
            component_p_values[component_name].append(value)
    
    # Compute min k and max p for each component
    results = {}
    
    print("Component-wise statistics across all 32 layers:")
    print("Component\t\t\t\tMin_K\t\tMax_P")
    print("-" * 80)
    
    # Get all unique component names
    all_components = set(component_k_values.keys()) | set(component_p_values.keys())
    
    for component in sorted(all_components):
        max_p = max(component_p_values[component]) if component in component_p_values else 0.0
        
        results[component] = {
            'max_p': max_p
        }
        
        print(f"{component:<40}\t{max_p:.6e}")
    
    # Find global minimum k and maximum p across all components
    all_max_p_values = [results[comp]['max_p'] for comp in results]
    
    global_max_p = max(all_max_p_values) if all_max_p_values else 0.0
    
    print(f"Global maximum p across all components: {global_max_p}")
    
    # Save processed statistics
    torch.save(results, f"{model_name}_component_worse_prob_analysis.pth")
    
    return {
        'global_max_p': global_max_p,
        'component_stats': results
    }

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)
    logger.info(f"args: {args}")

    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    # if quant_config['set_prefixed_tokens']:
    if quant_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
    else:
        prefixed_key_values = None
    prefix_len = len(quant_config.prefixed_tokens)
    # init quantized model
    config = AutoConfig.from_pretrained(args.quant_model_path,trust_remote_code=True)
    config._attn_implementation = "eager"
    config.output_attentions = True
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False,legacy=False,trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
    for name,layer in model.named_modules():
        if isinstance(layer,LlamaAttention):
            layer.prefix_len = prefix_len
    wrap_to_quant_model(model)
    # register on-line hadadamrd transformation
    if quant_config.down_online_had:
        register_online_had(model)
    # wrap rope for online_had and rope output capture
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name, 
                    config=model.config,
                    online_had=quant_config.qk_online_had)

    # init weight quantizer
    if quant_config.wbits < 16:
        logger.info('init weight quantizer')
        init_weight_quantizer(quant_config, model, minmax_init=False)

    # init input quantizer
    if quant_config.input_bits < 16:
        logger.info('init input quantizer')
        init_input_quantizer(quant_config, model,  minmax_init=False)

    # init kv quantizer
    if quant_config.v_bits < 16:
        logger.info('init v quantizer')
        init_v_quantizer(quant_config, model,  minmax_init=False)

    # if True:
    if quant_config.k_bits < 16:
        # consistently init for wrap rope 
        logger.info('init k quantizer')
        init_k_quantizer(quant_config, model,  minmax_init=False)
    
    if quant_config.s_bits < 16:
        logger.info('init s quantizer')
        init_s_quantizer(quant_config, model, minmax_init=False)

    # model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=args.quant_model_path,device_map=device_map,dtype=torch.float16)
    model.half()    # to make sure same evaluation results with main
    logger.info(model)
    # b_acc = compute_bacc(model)
    add_overflow_hook(model)
    # logger.info(f"b_acc: {b_acc}")
    model_name = "llama3"
    evaluate(model, tokenizer, prefixed_key_values,  args,logger)
    torch.save(stat, f"{model_name}_stat.pth")
    process_stat(model_name)
    compute_bound(model, model_name)
    ans = compute_worse_prob(model_name)
    print(ans)



if __name__ == "__main__":
    main()
    # model_name = "llama2"
    # compute_bound(model, model_name)
    # ans = compute_worse_prob(model_name)
    # print(ans)
    # a = torch.load("llama3_component_worse_prob_analysis.pth",weights_only=False)
    # print(a)

