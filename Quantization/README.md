# PrivLLM-Q
Official PyTorch implement for PrivLLM-Q quantization part. This repo is based on [PrefixQuant](https://github.com/MengzhaoChen/PrefixQuant).


## Installation
```
conda create -n prefixquant python==3.10

conda activate prefixquant

pip install -r requirements.txt
```
- Note that you must install the correct version of `transformers`
- If you cannot install `fast-hadamard-transform`, you can install it from source code.
  - git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
  - cd fast-hadamard-transform
  - pip install -e .
## Code Modification
In `YOUR_PATH/transformers/models/llama/modeling_llama.py`, add these code into `LlamaAttention` class, `forward` function.
``` python
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
# add the following code to quantize the attention weights
if hasattr(self, 's_quantizer') and self.s_quantizer is not None:
            prefix_len = self.prefix_len
            quantized_suffix = self.s_quantizer(attn_weights[:,:,:,prefix_len:])
            attn_weights = torch.cat([attn_weights[:,:,:,:prefix_len], quantized_suffix], dim=-1)
```

## Reproduce W4A4Q4S8KV4 Quantization Results
#### 1.Quantized the model
We provide an example command to quantized `Llama-3-8B` without fine-tuning:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path /opt/pretrained_models/Llama-3-8B \
--model_name Llama-3-8B \
--output_dir ./log/Llama-3-8B-w4a4q4s8kv4 \
--wbits 4 \
--input_bits 4 \
--input_mode static \
--v_bits 4 \
--k_bits 4 \
--s_bits 8 \
--kv_group_size 128 \
--kv_mode static \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--eval_batch_size 64 \
--save_quant_dir ./pre_quantized_models/Llama-3-8B-w4a4q4s8kv4
```
There are some useful information as follows:
- `--model_path` is the path to the pretrained model.
- `--save_quant_dir` is the path to save the quantized model.
#### 2.Evaluate the quantized model without clip
Change the code in `quantize/quantizer.py`, change `if True` on line 209 to `if self.quant_type == "weight":`; change `if True` on line 244 to `if self.quant_type == "weight":`. Comment out `x_int = x_int.clamp(self.qmin, self.qmax)` on line 264.

Then evaluate the quantized model:
```
CUDA_VISIBLE_DEVICES=5 python eval.py \
--quant_model ./pre_quantized_models/Llama-3-8B-w4a4q4s8kv4 \
--eval_batch_size 64 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
```
Then, you should get the results shown in our paper, for example:
```
[2025-08-08 00:05:01 root] (main.py 64): INFO Average Acc (with norm): 67.76%
```
## Check the probility of overflow without clip
Modify the code in `check_overflow.py`, change `model_name` to `llama2` or `llama3` to check the probility of overflow for `Llama-2-7B` or `Llama-3-8B`. Note that you should comment out the clip operator in `quantize/quantizer.py` as described before to get the same results as ours. Then run the following command:

```
CUDA_VISIBLE_DEVICES=5 python check_overflow.py \
--quant_model ./pre_quantized_models/Llama-3-8B-w4a4q4s8kv4 \
--eval_batch_size 64 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande

```
The results will contain the probility of overflow for each component. For example:
```
'model.layers.0.self_attn.q_proj': 0.0, 'model.layers.0.self_attn.q_proj_k': 82.63000982057005
```
where k represents the bound in the therom1 and theorm2 in our paper.
