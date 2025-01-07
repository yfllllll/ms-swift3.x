adapters=/data1/lyf/my_ms_swift/output/Qwen2-VL-7B-Instruct/7b_e10_n_aug/checkpoint-9160
mergeed=${adapters}-merged
output_dir=${adapters}-qwen7b-GPTQ-Int4
export OMP_NUM_THREADS=14 
export CUDA_VISIBLE_DEVICES=0 
export MAX_PIXELS=921600 
swift export \
    --adapters ${adapters} \
    --merge_lora true
swift export \
    --model ${mergeed} \
    --dataset ../output.jsonl \
    --quant_n_samples 128 \
    --quant_batch_size 1 \
    --max_length 2048 \
    --quant_method gptq \
    --quant_bits 4 \
    --output_dir ${output_dir}
