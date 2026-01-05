CUDA_VISIBLE_DEVICES=1 python eval.py \
--quant_model ./pre_quantized_models/Llama-3-8B-w4a4q4s8kv4-finetune \
--eval_batch_size 64 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande

