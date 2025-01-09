import os
import gradio as gr
from gradio_image_prompter import ImagePrompter
from PIL import Image, ImageDraw
import numpy as np
import secrets
import torch
import re
import gc
from PIL import ImageFont
# 初始化环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MAX_PIXELS'] = '921600'
os.environ["GRADIO_TEMP_DIR"] = "/data1/lyf/tmp"

# swift 模型加载相关模块
from swift.llm import InferEngine, PtEngine, InferRequest, RequestConfig
from swift.plugin import InferStats
from swift.utils import seed_everything

# 设置随机种子
seed_everything(42)

# 模型加载与初始化
last_model_checkpoint = '/data1/lyf/my_ms_swift/output/Qwen2-VL-7B-Instruct/v4-20250108-150848/checkpoint-4732'
print("Loading model...")

# 加载推理引擎
def initialize_engine_quant(model_id_or_path):
    engine_type = 'pt'
    if engine_type == 'pt':
        pt_engine_kwargs = {'max_batch_size': 8, 'torch_dtype': torch.float16}
        engine = PtEngine(model_id_or_path, **pt_engine_kwargs)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")
    return engine

# 初始化推理引擎

engine_type = 'pt' # 用户选择推理引擎
if('Int4') in last_model_checkpoint:
    engine = initialize_engine_quant(engine_type, last_model_checkpoint)# 用于加载量化过的模型
else:
    # Get model and template, and load LoRA weights.
    model_id_or_path = 'qwen/Qwen2-VL-7B-Instruct'
    engine = PtEngine(model_id_or_path, adapters=[last_model_checkpoint])
request_config = RequestConfig(max_tokens=4096, temperature=0)

# 清理 GPU 显存
def clear_cuda_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 绘制检测框

# 绘制检测框
def draw_boxes_on_image(image, boxes, color="red", font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size=20):
    """
    在图像上绘制检测框和标签，支持中文。
    
    Args:
    - image (numpy.ndarray): 输入的图像。
    - boxes (list): 包含检测框和标签的列表，格式为 [(class_name, [x1, y1, x2, y2]), ...]。
    - color (str): 框和标签文字的颜色。
    - font_path (str): 字体文件路径，需支持中文。
    - font_size (int): 字体大小。

    Returns:
    - Image: 带有检测框和标签的图像。
    """
    # 将图像转换为 PIL 格式
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    # 加载字体文件，确保支持中文
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    # 绘制每个检测框
    for class_name, box in boxes:
        x1, y1, x2, y2 = box
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签文字
        text = f"{class_name}"
        
        # 使用 textbbox 替代 textsize 获取文本大小
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_background = [x1, y1 - text_height, x1 + text_width, y1]  # 标签背景
        draw.rectangle(text_background, fill=color)  # 填充背景
        draw.text((x1, y1 - text_height), text, fill="white", font=font)  # 绘制文字

    return image_pil



# 裁剪图像区域
def crop_region(image_data, box):
    """裁剪指定的矩形区域"""
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image_data[y1:y2, x1:x2]
    return cropped_image

# 更新后的解析函数
def parse_response_boxes(response, scale_x, scale_y):
    """解析 response 中的矩形框并按比例还原坐标"""
    box_pattern = r"<\|object_ref_start\|\>(.*?)<\|object_ref_end\|\><\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    boxes = []
    matches = re.findall(box_pattern, response)
    for match in matches:
        class_name, x1, y1, x2, y2 = match
        x1 = int(int(x1) * scale_x)
        y1 = int(int(y1) * scale_y)
        x2 = int(int(x2) * scale_x)
        y2 = int(int(y2) * scale_y)
        boxes.append((class_name, [x1, y1, x2, y2]))
    return boxes

# 处理图像并推断
def process(image_input, full_image_or_region, category_name):
    image_data = image_input.get('image')
    points = image_input.get('points')

    # 保存临时目录
    image_dir = "/data1/lyf/tmp"
    os.makedirs(image_dir, exist_ok=True)

    image_local_path = []
    original_width, original_height = image_data.shape[1], image_data.shape[0]

    if full_image_or_region == "Full Image":
        image_filename = f"full_image_{secrets.token_hex(8)}.png"
        Image.fromarray(image_data).save(os.path.join(image_dir, image_filename))
        image_local_path.append(os.path.abspath(os.path.join(image_dir, image_filename)))
        scale_x = original_width / 1000
        scale_y = original_height / 1000

    elif full_image_or_region == "Drawn Regions" and points:
        # 解析用户绘制区域
        boxes = []
        for point in points:
            if point[2] == 2:  # 只处理绘制的矩形区域
                x1, y1, _, x2, y2, _ = point
                boxes.append([x1, y1, x2, y2])

        for idx, box in enumerate(boxes):
            cropped_image = crop_region(image_data, box)
            cropped_image_filename = f"region_{secrets.token_hex(8)}_{idx}.png"
            cropped_image_path = os.path.join(image_dir, cropped_image_filename)
            Image.fromarray(cropped_image).save(cropped_image_path)
            image_local_path.append(os.path.abspath(cropped_image_path))
            scale_x = (box[2] - box[0]) / 1000
            scale_y = (box[3] - box[1]) / 1000

    else:
        return "请在图像上绘制区域或选择整幅图像", "", ""

    # 构建推理请求
    query = f"<image>{category_name}"
    infer_request = InferRequest(
        messages=[{"role": "user", "content": query}],
        images=image_local_path
    )

    # 模型推理
    metric = InferStats()
    response = engine.infer([infer_request], request_config, metrics=[metric])
    response_text = response[0].choices[0].message.content

    # 解析检测框
    parsed_boxes = parse_response_boxes(response_text, scale_x, scale_y)

    # 绘制检测框
    final_boxes = []
    if parsed_boxes:
        for box in parsed_boxes:
            final_boxes.append(box)

        image_with_boxes = draw_boxes_on_image(image_data, final_boxes, color="blue")
    else:
        image_with_boxes = Image.fromarray(image_data)

    # 保存结果图像
    result_image_filename = f"result_image_{secrets.token_hex(8)}.png"
    result_image_path = os.path.join(image_dir, result_image_filename)
    image_with_boxes.save(result_image_path)

    clear_cuda_memory()
    return str(final_boxes), response_text, result_image_path

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        image_prompter = ImagePrompter(label="Draw on Image", scale=1)
        option = gr.Radio(choices=["Full Image", "Drawn Regions"], label="Select Input Type", value="Full Image")
        category_input = gr.Textbox(label="Enter Category Name", placeholder="Type category name here")
        bbox_output = gr.Textbox(label="Bounding Boxes", lines=10)
        response_output = gr.Textbox(label="Response", lines=3)
        result_image = gr.Image(label="Result Image", type="filepath")
        submit_button = gr.Button("提交")

    submit_button.click(fn=process, inputs=[image_prompter, option, category_input], outputs=[bbox_output, response_output, result_image])

demo.launch(share=False, server_name='0.0.0.0', server_port=8005)
