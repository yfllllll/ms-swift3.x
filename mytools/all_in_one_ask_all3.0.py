import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MAX_PIXELS'] = '921600'
import torch
import json
import re
import yaml
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
        # pt_engine_kwargs['torch_dtype'] =torch.float16
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
def process_folder(root_dir, folder, saved_folder='pred', engine=None, request_config=None):
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

    pred_folder = os.path.join(input_folder, saved_folder)
    os.makedirs(pred_folder, exist_ok=True)

    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_path in tqdm(image_files, desc=f"Processing folder: {folder}", unit="image"):
        process_image(image_path, class_names, pred_folder, engine, request_config)

    print(f"All predictions saved in: {pred_folder}")

def process_folders(root_dir, folder_list, saved_folder='pred', engine=None, request_config=None):
    # logger = setup_logger()
    for folder in folder_list:
        print(f"Processing folder: {folder}")
        # try:
        process_folder(root_dir, folder, saved_folder=saved_folder, engine=engine, request_config=request_config)
        # except Exception as e:
        #     logger.error(f"Error processing folder {folder}: {e}", exc_info=True)  # 记录错误和调用栈信息

# 主程序入口
if __name__ == "__main__":
    # 用户选择推理引擎
    engine_type = 'pt'
    model_id_or_path = '/data1/lyf/my_ms_swift/output/Qwen2-VL-7B-Instruct/7b_e10_agu/checkpoint-9160-qwen7b-GPTQ-Int4'

    # 初始化推理引擎
    engine = initialize_engine(engine_type, model_id_or_path)
    request_config = RequestConfig(max_tokens=256, temperature=0)

    # 设置路径
    root_dir = "/data1/lyf/datasets/VisDrone"
    folder_list = ["VisDrone2019-DET-test-dev"]
    saved_folder = 'pred_multi_engine'

    # 开始处理
    process_folders(root_dir, folder_list, saved_folder=saved_folder, engine=engine, request_config=request_config)

    print('Processing completed.')
