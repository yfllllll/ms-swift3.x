import os
import json
import re
import yaml
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import fiftyone as fo
from swift.llm import InferEngine, PtEngine, InferRequest, RequestConfig
from swift.plugin import InferStats

# 初始化推理引擎函数
def initialize_engine(engine_type='pt', model_id_or_path=None):
    if engine_type == 'pt':
        pt_engine_kwargs = {}
        pt_engine_kwargs['max_batch_size'] = 8
        engine = PtEngine(model_id_or_path, **pt_engine_kwargs)
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


# 保存检测结果为通用 JSON 格式
def save_detection_results(predictions, output_path, image_width, image_height):
    results = []
    for class_name, box in predictions:
        x1, y1, x2, y2 = box
        bounding_box = [
            x1 / image_width,
            y1 / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        ]
        results.append({
            "label": class_name,
            "bounding_box": bounding_box,
            "confidence": 1.0,  # 假设模型输出置信度为 1.0
        })
    # 写入 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# 处理每张图片
def process_image(image_path, class_names, output_dir, engine, request_config):
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    if os.path.exists(output_file):
        print(f"Skipping {image_path} (already processed)")
        return

    image = Image.open(image_path)
    image_width, image_height = image.size
    scale_x = image_width / 1000
    scale_y = image_height / 1000

    # 构建 query，一次性查询所有类别
    class_list_str = "，".join(class_names)
    query = f"<image>请检测图像中的以下类别：{class_list_str}"

    infer_request = InferRequest(
        messages=[{"role": "user", "content": query}],
        images=[image_path]
    )

    metric = InferStats()
    response = engine.infer([infer_request], request_config, metrics=[metric])
    response_text = response[0].choices[0].message.content

    predictions = parse_response_boxes(response_text, scale_x, scale_y)
    save_detection_results(predictions, output_file, image_width, image_height)
    print(f"Processed and saved: {image_path}")


# 处理单个文件夹
def process_folder(root_dir, folder, saved_folder='pred', engine=None, request_config=None, selection_param=1.0, seed=None, class_names=None):
    images_folder = os.path.join(root_dir, folder, 'images')
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    pred_folder = os.path.join(root_dir, folder, saved_folder)
    os.makedirs(pred_folder, exist_ok=True)

    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if isinstance(selection_param, float) and 0 < selection_param <= 1:
        image_count = int(len(image_files) * selection_param)
    elif isinstance(selection_param, int) and selection_param > 1:
        image_count = min(selection_param, len(image_files))
    else:
        raise ValueError("selection_param must be a float between 0 and 1 (percentage) or an integer greater than 1 (count)")

    selected_images = random.sample(image_files, image_count)

    for image_path in tqdm(selected_images, desc=f"Processing folder: {folder}", unit="image"):
        process_image(image_path, class_names, pred_folder, engine, request_config)

    print(f"All predictions saved in: {pred_folder}")


# 加载预测结果到 FiftyOne
def load_predictions_to_fiftyone(root_dir, folder, saved_folder='pred', dataset_name="predictions_dataset"):
    pred_folder = os.path.join(root_dir, folder, saved_folder)
    images_folder = os.path.join(root_dir, folder, 'images')

    dataset = fo.Dataset(name=dataset_name, overwrite=True)
    samples = []

    for json_file in os.listdir(pred_folder):
        if json_file.endswith(".json"):
            image_file = os.path.splitext(json_file)[0] + ".jpg"
            image_path = os.path.join(images_folder, image_file)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for {json_file}")
                continue

            with open(os.path.join(pred_folder, json_file), "r", encoding="utf-8") as f:
                detections_data = json.load(f)

            detections = [
                fo.Detection(
                    label=d["label"],
                    bounding_box=d["bounding_box"],
                    confidence=d.get("confidence", 1.0),
                )
                for d in detections_data
            ]

            sample = fo.Sample(filepath=image_path)
            sample["predictions"] = fo.Detections(detections=detections)
            samples.append(sample)

    dataset.add_samples(samples)
    session = fo.launch_app(dataset, address="0.0.0.0", port=8001)
    session.wait()


# 主程序入口
if __name__ == "__main__":
    engine_type = 'pt'
    model_id_or_path = '/data1/lyf/my_ms_swift/output/Qwen2-VL-7B-Instruct/7b_e10_agu/checkpoint-9160-qwen7b-GPTQ-Int4'
    engine = initialize_engine(engine_type, model_id_or_path)
    request_config = RequestConfig(max_tokens=256, temperature=0)

    root_dir = "/data1/lyf/datasets/VisDrone"
    folder_list = ["VisDrone2019-DET-test-dev"]
    saved_folder = 'pred_non_yolo'
    selection_param = 100
    seed = 42
    class_names = ["行人", "车辆", "自行车"]

    for folder in folder_list:
        process_folder(root_dir, folder, saved_folder=saved_folder, engine=engine, request_config=request_config, selection_param=selection_param, seed=seed, class_names=class_names)

    load_predictions_to_fiftyone(root_dir, folder_list[0], saved_folder=saved_folder, dataset_name="non_yolo_predictions")
