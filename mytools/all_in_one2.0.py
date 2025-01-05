'''
相较于all_in_one 增加了中断继续预测的能力
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MAX_PIXELS']= '921600'
import json
import re
import yaml
from tqdm import tqdm
from PIL import Image
import torch
import fiftyone as fo
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from swift.tuners import Swift


# 模型初始化
model_id_or_path = '/data1/lyf/ms-swift3.x/output/qwen2-vl-7b-instruct/all/checkpoint-42719-gptq-int4'
model_type = ModelType.qwen2_vl_7b_instruct
template_type = get_default_template_type(model_type)
print(f"template_type: {template_type}")

model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_id_or_path=model_id_or_path, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
print("Loading model...")
seed_everything(42)

# 解析 response 中的 bbox
def parse_response_boxes(response, scale_x, scale_y):
    box_pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    boxes = []
    matches = re.findall(box_pattern, response)
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        boxes.append([x1, y1, x2, y2])
    return boxes


def load_yolo_file(class_list_path):
    with open(class_list_path, 'r', encoding='utf-8') as f:
        class_list = [line.strip() for line in f.readlines() if line.strip()]
    return class_list

def load_yolo_labels(label_path, classes, pred=False):
    detections = []
    lines = load_yolo_file(label_path)
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        label = classes[class_id]
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])
        bounding_box = [
            x_center - bbox_width / 2,
            y_center - bbox_height / 2,
            bbox_width,
            bbox_height,
        ]
        if pred==True:
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=bounding_box,
                    confidence=1
                )
            )
        else:
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=bounding_box,
                )
            )
            
    return detections


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

# 处理每张图片
def process_image(image_path, class_names, output_dir, processed_images):
    try:
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

        # 遍历每个类别
        for class_id, class_name in class_names.items():
            query = f"找到{class_name}"  # 构建 query
            response, _ = inference(model, template, query, images=[image_path])  # 模型推理

            # 解析检测结果
            pred_boxes = parse_response_boxes(response, scale_x, scale_y)

            # 保存为 YOLO 格式
            save_yolo_format(pred_boxes, output_file, class_id, image_width, image_height, append=True)

        print(f"Processed and saved: {image_path}")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# 处理单个文件夹
def process_folder(root_dir, folder, saved_folder='pred'):
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
        process_image(image_path, class_names, pred_folder, processed_images=None)

    print(f"All predictions saved in: {pred_folder}")

# 处理多个文件夹
def process_folders(root_dir, folder_list, saved_folder='pred'):
    for folder in folder_list:
        print(f"Processing folder: {folder}")
        try:
            process_folder(root_dir, folder, saved_folder=saved_folder)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

def evaluate_predictions(dataset_name, folder_list, root_dir, iou_threshold=0.5, saved_folder='pred'):
    dataset = fo.Dataset(name=dataset_name, overwrite=True)
    samples = []
    class_names = None

    for folder in folder_list:
        print(f"Processing folder: {folder}")
        pred_dir = os.path.join(root_dir, folder, saved_folder)
        gt_dir = os.path.join(root_dir, folder, "labels")
        images_dir = os.path.join(root_dir, folder, "images")

        if not os.path.exists(pred_dir) or not os.path.exists(gt_dir) or not os.path.exists(images_dir):
            print(f"Warning: Missing required folders in {folder}. Skipping...")
            continue

        yaml_path = os.path.join(root_dir, folder, "dataset_zn.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
                class_names = data_config.get("names", [])
        else:
            print(f"Warning: dataset.yaml not found in {folder}. Using default class names.")
            continue

        for file_name in os.listdir(gt_dir):
            if file_name.endswith(".txt"):
                image_name = file_name.replace(".txt", ".jpg")
                image_path = os.path.join(images_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Warning: Image not found for {file_name}")
                    continue

                gt_label_path = os.path.join(gt_dir, file_name)
                gt_detections = load_yolo_labels(gt_label_path, class_names)

                pred_label_path = os.path.join(pred_dir, file_name)
                pred_detections = load_yolo_labels(pred_label_path, class_names, pred=True)

                sample = fo.Sample(filepath=image_path)
                sample["ground_truth"] = fo.Detections(detections=gt_detections)
                sample["predictions"] = fo.Detections(detections=pred_detections)
                samples.append(sample)

    dataset.add_samples(samples)

    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="evaluation",
        iou=iou_threshold,
        compute_mAP=True,
    )

    print("\nEvaluation Report:")
    results.print_report()

    session = fo.launch_app(dataset)
    session.wait()

    return results


# 调用并继续处理
root_dir = "/data1/lyf/data/val"
folder_list = ["life_vest_yolo"]
saved_folder = 'pred7b'

process_folders(root_dir, folder_list, saved_folder=saved_folder)

results = evaluate_predictions(
    dataset_name="my_combined_dataset",
    folder_list=folder_list,
    root_dir=root_dir,
    iou_threshold=0.5,
    saved_folder=saved_folder
)

print('done')
