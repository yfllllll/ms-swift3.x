NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAX_PIXELS=921600 \
swift sft \
  --model qwen/Qwen2-VL-2B-Instruct \
  --eval_steps 3000 \
  --dataloader_num_workers 1 \
  --dataset dataset/fromVary_3.0/Tower_bigdata_train.jsonl \
            dataset/fromVary_3.0/Tower_data_1_train.jsonl \
  --deepspeed default-zero3 

  
  
  
#dataset/fromVary/alg_base_Cap_train.jsonl,dataset/fromVary/Tower_bigdata_cap_train.jsonl,dataset/fromVary/Tower_bigdata_regionCap_train.jsonl, 
#dataset/fromVary/alg_base_Cap_val.jsonl,dataset/fromVary/Tower_bigdata_cap_val.jsonl,dataset/fromVary/Tower_bigdata_regionCap_val.jsonl,
