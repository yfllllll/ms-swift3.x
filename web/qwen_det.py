import gradio as gr
from gradio_image_prompter import ImagePrompter
from PIL import Image, ImageDraw
import numpy as np
import secrets
import os
from swift.llm import get_model_tokenizer, get_template, inference, ModelType, get_default_template_type, inference_stream
from swift.utils import seed_everything
from swift.tuners import Swift
import torch
import re

os.environ["GRADIO_TEMP_DIR"] = "/data1/lyf/tmp"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 初始化模型和配置
ckpt_dir = '/ms-swift/output/qwen2-vl-7b-instruct/v29-20241114-103453/checkpoint-300/'
model_id_or_path = None
model_type = ModelType.qwen2_vl_7b_instruct
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_id_or_path=model_id_or_path, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
seed_everything(42)

def parse_visual_prompt(points):
    boxes = []
    for point in points:
        if point[2] == 2:
            x1, y1, _, x2, y2, _ = point
            boxes.append([x1, y1, x2, y2])
    return boxes

def draw_boxes_on_image(image, boxes, color="red"):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return image_pil

def crop_region(image_data, box):
    """裁剪指定的矩形区域"""
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image_data[y1:y2, x1:x2]
    return cropped_image

def parse_response_boxes(response, scale_x, scale_y):
    """解析 response 中的矩形框并按比例还原坐标"""
    box_pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    boxes = []
    matches = re.findall(box_pattern, response)
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        # 将坐标还原到原始图像的尺寸
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        boxes.append([x1, y1, x2, y2])
    return boxes

def process(image_input, full_image_or_region):
    image_data = image_input.get('image')
    points = image_input.get('points')
    boxes = parse_visual_prompt(points)
    boxes_str = str(boxes) if boxes else ''

    image_dir = "/data1/lyf/tmp"
    os.makedirs(image_dir, exist_ok=True)

    original_width, original_height = image_data.shape[1], image_data.shape[0]
    scale_x = original_width / 1000
    scale_y = original_height / 1000

    image_local_path = []
    if full_image_or_region == "Full Image":
        # 使用整幅图像
        image_filename = f"full_image_{secrets.token_hex(8)}.png"
        Image.fromarray(image_data).save(os.path.join(image_dir, image_filename))
        image_local_path.append(os.path.abspath(os.path.join(image_dir, image_filename)))

    elif full_image_or_region == "Drawn Regions" and boxes:
        # 使用绘制区域
        for idx, box in enumerate(boxes):
            cropped_image = crop_region(image_data, box)
            cropped_image_filename = f"region_{secrets.token_hex(8)}_{idx}.png"
            cropped_image_path = os.path.join(image_dir, cropped_image_filename)
            Image.fromarray(cropped_image).save(cropped_image_path)
            image_local_path.append(os.path.abspath(cropped_image_path))

    else:
        return "请在图像上绘制区域或选择整幅图像", ""

    query = '<image>Find Tricycle'

    # 进行模型推理
    response, history = inference(model, template, query, images=image_local_path)

    # 解析 response 中的矩形框，按比例还原到原始图像尺寸
    response_boxes = parse_response_boxes(response, scale_x, scale_y)
    if response_boxes:
        image_with_boxes = draw_boxes_on_image(image_data, response_boxes, color="blue")
    else:
        image_with_boxes = Image.fromarray(image_data)

    # 将带有框的图像保存到临时文件夹并返回路径
    result_image_filename = f"result_image_{secrets.token_hex(8)}.png"
    result_image_path = os.path.join(image_dir, result_image_filename)
    image_with_boxes.save(result_image_path)

    return boxes_str, response, result_image_path

with gr.Blocks() as demo:
    with gr.Row():
        image_prompter = ImagePrompter(label="Draw on Image", scale=1)
        option = gr.Radio(choices=["Full Image", "Drawn Regions"], label="Select Input Type", value="Full Image")
        bbox_output = gr.Textbox(label="Bounding Boxes", lines=10)
    
    response_output = gr.Textbox(label="Response", lines=10)
    result_image = gr.Image(label="Result Image", type="filepath")
    submit_button = gr.Button("提交")

    # 调用 process 函数，返回 bbox_output, response_output, result_image
    submit_button.click(fn=process, inputs=[image_prompter, option], outputs=[bbox_output, response_output, result_image])

demo.launch(share=False, server_name='0.0.0.0', server_port=8004)
