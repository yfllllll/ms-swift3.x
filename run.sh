

CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=921600 swift sft \
  --model qwen/Qwen2-VL-7B-Instruct \
  --train_type lora \
  --dataset output.jsonl 
