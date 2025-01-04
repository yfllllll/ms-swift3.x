CUDA_VISIBLE_DEVICES=7 \
MAX_PIXELS=921600 \
swift export \
	--ckpt_dir '/data1/lyf/ms-swift/output/qwen2-vl-2b-instruct/v6-20241206-123036/checkpoint-42719' \
	--merge_lora true \
	--quant_bits 4 \
	--dataset /data1/lyf/ms-swift/dataset/fromVary/Tower_bigdata_train.jsonl \
			/data1/lyf/ms-swift/dataset/fromVary/Tower_data_1_train.jsonl \
			/data1/lyf/formate_t/coco_format/boat/train.jsonl \
			/data1/lyf/formate_t/coco_format/cheren/train.jsonl \
			/data1/lyf/formate_t/coco_format/jxsg/train.jsonl \
			/data1/lyf/formate_t/coco_format/laji/train.jsonl \
			/data1/lyf/formate_t/coco_format/smpfw/train.jsonl \
			/data1/lyf/formate_t/coco_format/swim/train.jsonl \
			/data1/lyf/formate_t/coco_format/water/train.jsonl \
            /data1/lyf/formate_t/coco_format/yanhuo/train.jsonl \
			/data1/lyf/ms-swift/dataset/fromVary/alg_base_Cap_train.jsonl \
			/data1/lyf/ms-swift/dataset/fromVary/Tower_bigdata_cap_train.jsonl \
			/data1/lyf/ms-swift/dataset/fromVary/Tower_bigdata_regionCap_train.jsonl \
	--quant_method gptq
