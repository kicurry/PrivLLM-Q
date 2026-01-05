### PrefixQuant
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path pretrained_models/Llama-2-7b-hf \
--model_name Llama-2-7b-hf \
--output_dir ./log/Llama-2-7b-hf-w4a4q4s8kv4 \
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
--eval_batch_size 64 \
--save_quant_dir ./pre_quantized_models/Llama-2-7b-hf-w4a4q4s8kv4
# --eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \