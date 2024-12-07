CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=921600 swift sft \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path qwen/Qwen2-VL-7B-Instruct \
  --sft_type lora \
  --dataset train.jsonl \
  --val_dataset val.jsonl \
  #--dataset refcoco-unofficial-grounding#20000 \
  --deepspeed default-zero3
