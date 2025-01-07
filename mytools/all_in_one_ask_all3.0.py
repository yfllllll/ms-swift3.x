import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MAX_PIXELS'] = '921600'
import torch
import json
import re
import yaml
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import fiftyone as fo
from swift.llm import InferEngine, PtEngine, VllmEngine, LmdeployEngine, InferRequest, RequestConfig
from swift.plugin import InferStats

# YOLO 格式保存函数
def save_yolo_format(predictions, output_path, class_id, image_width, image_height, append=False):
    mode = 'a' if append else 'w'  # 'a' for append, 'w' for overwrite
    with open(output_path, mode) as f:
        for box in predictions:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2.0 / image_width
            y_center = (y1 + y2) / 2.0 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
# 初始化推理引擎函数
def initialize_engine(engine_type='pt', model_id_or_path=None):
    if engine_type == 'pt':
        # 初始化参数字典
        pt_engine_kwargs = {}
        
        # 检查 model_id_or_path 是否是一个本地路径
        
        if os.path.isdir(model_id_or_path):
            args_json_path = os.path.join(model_id_or_path, 'args.json')
            # quantize_config_path = os.path.join(model_id_or_path, 'quantize_config.json')
            if os.path.exists(args_json_path):
                # 尝试读取 args.json 文件
                try:
                    with open(args_json_path, 'r', encoding='utf-8') as f:
                        args = json.load(f)
                        # 获取 model_type 字段并加入到参数字典
                        if 'model_type' in args:
                            pt_engine_kwargs['model_type'] = args['model_type']
                        else:
                            print(f"Warning: 'model_type' field not found in {args_json_path}")
                except Exception as e:
                    print(f"Error reading {args_json_path}: {e}")
            else:
                print(f"Warning: args.json not found in {model_id_or_path}")
            # if os.path.exists(quantize_config_path):
            #     # 尝试读取 args.json 文件
            #     try:
            #         with open(quantize_config_path, 'r', encoding='utf-8') as f:
            #             args = json.load(f)
            #             # 获取 model_type 字段并加入到参数字典
            #             pt_engine_kwargs['quantization_config'] = args
            #     except Exception as e:
            #         print(f"Error reading {quantize_config_path}: {e}")
            # else:
            #     print(f"Warning: args.json not found in {model_id_or_path}")
        # 添加其他 PtEngine 参数
        pt_engine_kwargs['max_batch_size'] = 8

        pt_engine_kwargs['torch_dtype'] =torch.float16

        # 初始化 PtEngine
        engine = PtEngine(model_id_or_path, **pt_engine_kwargs)

    elif engine_type == 'vllm':
        engine = VllmEngine(model_id_or_path, max_model_len=32768, limit_mm_per_prompt={'image': 5, 'video': 2})
    elif engine_type == 'lmdeploy':
        engine = LmdeployEngine(model_id_or_path, vision_batch_size=8)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")
    
    return engine


# 解析 response 中的 bbox 和类别
def parse_response_boxes(response, scale_x, scale_y):
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


# 处理每张图片
def process_image(image_path, class_names, output_dir, engine, request_config):
    # 检查该图片是否已经处理过
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return
    
    # 加载图像
    image = Image.open(image_path)
    image_width, image_height = image.size
    scale_x = image_width / 1000  # Assuming scaled width is 1000
    scale_y = image_height / 1000

    # 构建 query，一次性询问所有类别
    class_list_str = "，".join(class_names.values())
    query = f"<image>请检测图像中的以下类别：{class_list_str}"  # 构建 query

    # 构建推理请求
    infer_request = InferRequest(     
            messages=[{"role": "user", "content": query}],
            images=[image_path]
    )

    # 模型推理
    metric = InferStats()
    response = engine.infer([infer_request], request_config, metrics=[metric])
    response_text = response[0].choices[0].message.content

    # 解析检测结果
    parsed_boxes = parse_response_boxes(response_text, scale_x, scale_y)

    # 保存为 YOLO 格式
    for class_name, box in parsed_boxes:
        class_id = list(class_names.values()).index(class_name)  # 获取类别对应的 ID
        save_yolo_format([box], output_file, class_id, image_width, image_height, append=True)

    print(f"Processed and saved: {image_path}")


# 处理单个文件夹
# 修改后的处理单个文件夹函数，支持传入单一参数，按百分比或影像个数处理
def process_folder(root_dir, folder, saved_folder='pred', engine=None, request_config=None, selection_param=1.0, seed=None):
    input_folder = os.path.join(root_dir, folder)
    
    yaml_files = ['data_zn.yaml', 'dataset_zn.yaml']
    yaml_path = next((os.path.join(input_folder, f) for f in yaml_files if os.path.exists(os.path.join(input_folder, f))), None)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data_zn.yaml not found in {input_folder}")
    
    with open(yaml_path, 'r') as yaml_file:
        data_info = yaml.safe_load(yaml_file)
    
    class_names = data_info['names']  # 类别名称
    images_folder = os.path.join(input_folder, 'images')
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    # 获取所有图片文件
    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 处理随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # 确定处理的图片数量
    if isinstance(selection_param, float) and 0 < selection_param <= 1:
        # 如果是浮点数且在 (0, 1] 之间，按百分比处理
        image_count = int(len(image_files) * selection_param)
    elif isinstance(selection_param, int) and selection_param > 1:
        # 如果是整数且大于 1，按影像个数处理
        image_count = min(selection_param, len(image_files))
    else:
        raise ValueError("selection_param must be a float between 0 and 1 (percentage) or an integer greater than 1 (count)")

    selected_images = random.sample(image_files, image_count)
    
    pred_folder = os.path.join(input_folder, saved_folder)
    os.makedirs(pred_folder, exist_ok=True)

    # 逐个处理选中的图像
    for image_path in tqdm(selected_images, desc=f"Processing folder: {folder}", unit="image"):
        process_image(image_path, class_names, pred_folder, engine, request_config)

    print(f"All predictions saved in: {pred_folder}")


# 修改后的处理文件夹列表函数，支持灵活设置百分比或影像个数
def process_folders(root_dir, folder_list, saved_folder='pred', engine=None, request_config=None, selection_param=1.0, seed=None):
    for i, folder in enumerate(folder_list):
        print(f"Processing folder: {folder}")

        # 如果传入的是一个列表（每个数据集不同的参数）
        if isinstance(selection_param, list):
            folder_selection_param = selection_param[i]  # 获取该数据集的参数
        else:
            folder_selection_param = selection_param  # 对所有数据集都应用相同的参数
        
        try:
            process_folder(root_dir, folder, saved_folder=saved_folder, engine=engine, request_config=request_config, 
                           selection_param=folder_selection_param, seed=seed)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
# 主程序入口
if __name__ == "__main__":
    # 用户选择推理引擎
    engine_type = 'pt'
    model_id_or_path = '/data1/lyf/my_ms_swift/output/Qwen2-VL-7B-Instruct/7b_e10_n_aug/checkpoint-9160-qwen7b-GPTQ-Int4'
    # 初始化推理引擎
    engine = initialize_engine(engine_type, model_id_or_path)
    request_config = RequestConfig(max_tokens=256, temperature=0)
    # 设置路径
    root_dir = "/data1/lyf/datasets/VisDrone"
    folder_list = ["VisDrone2019-DET-test-dev"]
    saved_folder = 'pred_multi_engine_n_aug'
    selection_param = 100  # 对第一个数据集处理50%的图片，第二个数据集处理100张图片     # selection_param = 0.25  # 或者对所有数据集处理25%的图片 
     # 设置随机种子     
    seed = 42 
     # 开始处理     
    process_folders(root_dir, folder_list, saved_folder=saved_folder, engine=engine, request_config=request_config, selection_param=selection_param, seed=seed) 
    print('Processing completed.')
