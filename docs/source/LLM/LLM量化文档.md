# LLM量化文档
swift使用awq技术对模型进行量化. 该量化技术支持vllm进行加速推理.


## 目录
- [环境准备](#环境准备)
- [原始模型](#原始模型)
- [微调后模型](#微调后模型)
- [推送模型](#推送模型)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]

# autoawq和cuda版本有对应关系，请按照`https://github.com/casper-hansen/AutoAWQ`选择版本
pip install autoawq -U

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 原始模型
这里展示对qwen1half-7b-chat进行awq量化.
```bash
# awq-int4量化 (使用A100大约需要18分钟, 显存占用: 12GB)
# 如果出现量化的时候OOM, 可以适度降低`--quant_n_samples`(默认256)和`--quant_seqlen`(默认2048).

# 使用`ms-bench-mini`作为量化数据集
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset ms-bench-mini

# 使用自定义量化数据集 (`--custom_val_dataset_path`参数不进行使用)
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --custom_train_dataset_path xxx.jsonl

# 推理 swift量化产生的模型
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat --model_id_or_path qwen1half-7b-chat-int4
# 推理 原始模型
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
# 推理 qwen官方量化的awq模型
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat --model_id_or_path qwen/Qwen1.5-7B-Chat-AWQ
```


效果对比:
```python
# swift量化产生的模型
"""
<<< 你好
你好！有什么问题我可以帮助你吗？
--------------------------------------------------
<<< 2000年是闰年嘛？
是的，2000年是闰年。闰年规则是：普通年份能被4整除但不能被100整除，或者能被400整除的年份就是闰年。2000年满足后者，所以按照公历是闰年。
--------------------------------------------------
<<< 15869+587=?
15869 + 587 = 16456
--------------------------------------------------
<<< 浙江的省会在哪
浙江省的省会是杭州市。
--------------------------------------------------
<<< 这有什么好吃的
浙江美食非常丰富，以下列举一些具有代表性的：

1. **杭州菜**：如西湖醋鱼、东坡肉、龙井虾仁、叫化童鸡等，清淡而精致。
2. **宁波菜**：如红烧肉、汤圆、水磨年糕、宁波汤包等，口味鲜美。
3. **温州菜**：温州海鲜如蝤蛑、蝤蛑炒年糕、瓯菜，口味偏咸鲜。
4. **绍兴菜**：如东坡肉、霉干菜扣肉、绍兴黄酒等，酒香浓郁。
5. **嘉兴粽子**：如嘉兴肉粽，特别是五芳斋的粽子，口感软糯。
6. **金华火腿**：浙江特产，以金华地区最为著名，口感醇厚。
7. **浙东土菜**：如农家小炒、野菜、腌菜等，原汁原味。

各地还有许多特色小吃，如杭州的知味观、宋城的宋嫂鱼羹，以及各种小吃街如河坊街、西湖醋鱼一条街等，都值得一尝。如果你有具体地方或口味偏好，可以告诉我，我可以给出更具体的建议。
"""

# 原始模型
"""
<<< 你好
你好！有什么问题我可以帮助你吗？
--------------------------------------------------
<<< 2000年是闰年嘛？
是的，2000年是闰年。根据格里高利历（公历），闰年的规则是：普通年份能被4整除但不能被100整除，或者能被400整除的年份都是闰年。2000年满足后者，所以是闰年。
--------------------------------------------------
<<< 15869+587=?
15869 + 587 = 16456
--------------------------------------------------
<<< 浙江的省会在哪
浙江省的省会是杭州市。
--------------------------------------------------
<<< 这有什么好吃的
浙江的美食非常丰富，以下列举一些具有代表性的：

1. **杭州菜**：如西湖醋鱼、东坡肉、龙井虾仁、叫化童鸡等，清淡鲜美，注重原汁原味。
2. **宁波菜**：如宁波汤圆、红烧肉、海鲜类，如宁波海鲜面、清蒸河鳗等。
3. **绍兴菜**：如霉干菜扣肉、茴香豆、醉排骨，特色是酱香浓郁。
4. **温州菜**：如温州鱼丸、白斩鸡、楠溪江三鲜，口味偏咸鲜。
5. **嘉兴粽子**：特别是嘉兴五芳斋的粽子，闻名全国，甜咸皆有。
6. **金华火腿**：浙江名特产，口感鲜美，营养丰富。
7. **浙东土菜**：如东阳火腿、嵊州菜、台州海鲜等，地方特色鲜明。

当然，浙江各地还有许多特色小吃，如衢州的鸭头、湖州的粽子、舟山的海鲜等，你可以根据自己的口味选择。如果你需要更具体的推荐，可以告诉我你对哪种类型或者哪个地方的美食感兴趣。
"""

# qwen官方量化的awq模型
"""
<<< 你好
你好！有什么问题我可以帮助你吗？
--------------------------------------------------
<<< 2000年是闰年嘛？
是的，2000年是闰年。闰年的规则是：普通年份能被4整除的是闰年，但如果这个年份能被100整除，那么它不是闰年；但是，如果这个年份能被400整除，那么它仍然是闰年。2000年能被400整除，所以它是闰年。
--------------------------------------------------
<<< 15869+587=?
15869 + 587 = 16456
--------------------------------------------------
<<< 浙江的省会在哪
浙江省的省会是杭州市。
--------------------------------------------------
<<< 这有什么好吃的
浙江的美食非常丰富，以下是一些具有代表性的：

1. 杭州菜：杭州菜，又称浙菜，是中国八大菜系之一，以清鲜、酥嫩、原汁原味而闻名，如西湖醋鱼、龙井虾仁、叫化童鸡等。

2. 宁波汤圆：宁波汤圆是宁波的传统小吃，皮薄馅大，甜而不腻，有芝麻、豆沙、五仁等多种口味。

3. 温州瓯菜：温州菜以海鲜为主，口味偏咸鲜，如海鲜烩饭、海鲜面等。

4. 金华火腿：浙江金华的火腿以其独特的制作工艺和口感闻名，是馈赠亲友的佳品。

5. 衢州酥饼：衢州酥饼是衢州地区的特色小吃，酥脆可口，馅料多样。

6. 舟山海鲜：浙江沿海地区，海鲜种类丰富，如梭子蟹、海螺、带鱼等。

7. 东阳黑猪肉：东阳黑猪肉质鲜嫩，口感好，是浙江特色美食。

当然，浙江各地还有许多其他美食，如绍兴黄酒、嘉兴粽子、嵊州年糕等，你可以根据自己的口味去尝试。
"""
```


## 微调后模型

假设你使用lora微调了qwen1half-4b-chat, 模型权重目录为: `output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx`.

**Merge-LoRA & 量化**
```shell
# 使用`ms-bench-mini`作为量化数据集
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx' \
    --merge_lora true --quant_bits 4 \
    --dataset ms-bench-mini

# 使用微调时使用的数据集作为量化数据集
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx' \
    --merge_lora true --quant_bits 4 \
    --load_dataset_config true
```

**推理量化后模型**
```shell
# awq量化模型支持vllm推理加速. 也支持模型部署.
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged-int4'
```

**部署量化后模型**

服务端:

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged-int4'
```

测试:
```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen1half-4b-chat",
"messages": [{"role": "user", "content": "晚上睡不着觉怎么办？"}],
"max_tokens": 256,
"temperature": 0
}'
```


## 推送模型
假设你使用lora微调了qwen1half-4b-chat, 模型权重目录为: `output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx`.

```shell
# 推送lora增量模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>'

# 推送merged模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>' \
    --merge_lora true

# 推送量化后模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>' \
    --merge_lora true \
    --quant_bits 4
```
