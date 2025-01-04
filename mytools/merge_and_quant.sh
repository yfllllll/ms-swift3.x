adapters=/data1/lyf/my_ms_swift/output/Qwen2-VL-7B-Instruct/7b_agu/checkpoint-916

swift export \
    --adapters ${adapters} \
    --merge_lora true

# OMP_NUM_THREADS=14 \
# CUDA_VISIBLE_DEVICES=0 \
# swift export \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
#               'AI-ModelScope/alpaca-gpt4-data-en#500' \
#     --quant_n_samples 128 \
#     --quant_batch_size 1 \
#     --max_length 2048 \
#     --quant_method gptq \
#     --quant_bits 4 \
#     --output_dir Qwen2.5-1.5B-Instruct-GPTQ-Int4
