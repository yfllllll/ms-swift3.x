NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
MAX_PIXELS=921600 \
swift sft \
  --model qwen/Qwen2-VL-7B-Instruct \
  --eval_steps 3000 \
  --deepspeed zero3 \
  --save_steps 3000 \
  --num_train_epochs 10 \
  --strict False \
  --per_device_train_batch_size 1 \
  --dataset train.jsonl \
  --val_dataset val.jsonl


  
  
  
#dataset/fromVary/alg_base_Cap_train.jsonl,dataset/fromVary/Tower_bigdata_cap_train.jsonl,dataset/fromVary/Tower_bigdata_regionCap_train.jsonl, 
#dataset/fromVary/alg_base_Cap_val.jsonl,dataset/fromVary/Tower_bigdata_cap_val.jsonl,dataset/fromVary/Tower_bigdata_regionCap_val.jsonl,
