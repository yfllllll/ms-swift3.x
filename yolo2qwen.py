import os
import yaml
import json
import random
from PIL import Image

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def yolo_to_bbox(yolo_data, img_width, img_height, normalize_to_1000=False):
    object_data = {}

    for item in yolo_data:
        label, x_center, y_center, bbox_width, bbox_height = item
        x_min = int((x_center - bbox_width / 2) * img_width)
        y_min = int((y_center - bbox_height / 2) * img_height)
        x_max = int((x_center + bbox_width / 2) * img_width)
        y_max = int((y_center + bbox_height / 2) * img_height)

        if normalize_to_1000:
            x_min = int(x_min / img_width * 1000)
            y_min = int(y_min / img_height * 1000)
            x_max = int(x_max / img_width * 1000)
            y_max = int(y_max / img_height * 1000)
            bbox_type = "norm_1000"
        else:
            bbox_type = "real"

        if label not in object_data:
            object_data[label] = {
                "caption": label,
                "bbox": [],
                "bbox_type": bbox_type,
                "image": 0
            }
        
        object_data[label]["bbox"].append([x_min, y_min, x_max, y_max])
    
    return list(object_data.values())

def process_yolo_files(folder_path, normalize_to_1000=False):
    yaml_path = os.path.join(folder_path, "data.yaml") if os.path.exists(os.path.join(folder_path, "data.yaml")) else os.path.join(folder_path, "dataset.yaml")
    data_config = load_yaml(yaml_path)
    class_names = data_config["names"]

    labels_folder = os.path.join(folder_path, "labels")
    images_folder = os.path.join(folder_path, "images")
    jsonl_data = []
    
    for label_file in os.listdir(labels_folder):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_folder, label_file)
        image_filename = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(image_path):
            continue
        
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        with open(label_path, "r") as f:
            yolo_data = [list(map(float, line.strip().split())) for line in f.readlines()]
            yolo_data = [[class_names[int(item[0])]] + item[1:] for item in yolo_data]

        objects = yolo_to_bbox(yolo_data, img_width, img_height, normalize_to_1000)

        jsonl_data.append({
            "query": "Find <ref-object>",
            "response": "<bbox>",
            "images": [image_path],
            "objects": objects
        })

    return jsonl_data

def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data

# Example usage
folder_path = "/316/"
normalize_to_1000 = False
train_ratio = 0.8  # Training set ratio

converted_data = process_yolo_files(folder_path, normalize_to_1000)

# Split the data
train_data, val_data = split_data(converted_data, train_ratio)

# Save train.jsonl and val.jsonl
with open("train.jsonl", "w") as f_train, open("val.jsonl", "w") as f_val:
    for item in train_data:
        f_train.write(json.dumps(item) + "\n")
    for item in val_data:
        f_val.write(json.dumps(item) + "\n")

print("Train and validation split complete!")
