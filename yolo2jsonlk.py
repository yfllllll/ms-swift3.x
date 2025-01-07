import os
import yaml
import json
import random

def get_classes_from_yaml(yaml_path):
    """
    从dataset.yaml文件中解析出类别名列表
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    classes = [v for k, v in data.get('names', {}).items()]
    return classes

def get_image_label_pairs(image_dir, label_dir):
    """
    获取图像和标注文件路径对
    """
    image_extensions = ('.jpg', '.png', '.jpeg')  # 支持的图像文件扩展名
    label_extension = '.txt'
    image_label_pairs = []

    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    for image_file in image_files:
        image_file_name, _ = os.path.splitext(image_file)
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file_name + label_extension)
        if os.path.exists(label_path):
            image_label_pairs.append((image_path, label_path))

    return image_label_pairs

def write_to_jsonl(image_label_pairs, classes, output_path):
    """
    将图像和标注文件路径以及类别名列表按要求格式写入jsonl文件
    """
    with open(output_path, 'w') as jsonl_file:
        for image_path, label_path in image_label_pairs:
            record = {
                "image": os.path.abspath(image_path),
                "label_path": os.path.abspath(label_path),
                "class_names": classes,
                "yolo": 1
            }
            jsonl_file.write(json.dumps(record) + '\n')

def split_data(image_label_pairs, train_ratio=0.8):
    """
    按比例将数据分为训练集和验证集
    """
    random.shuffle(image_label_pairs)
    split_index = int(len(image_label_pairs) * train_ratio)
    train_pairs = image_label_pairs[:split_index]
    val_pairs = image_label_pairs[split_index:]
    return train_pairs, val_pairs

if __name__ == "__main__":
    dataset_dir = "/ultralytics-main/datasets/VisDrone/VisDrone2019-DET-train"  # 替换为你的数据集实际目录路径
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    classes = get_classes_from_yaml(yaml_path)

    # 检查是否有子文件夹 train、val、test
    subfolders = ['train', 'val', 'test']
    subfolder_structure = {subfolder: os.path.exists(os.path.join(dataset_dir, "images", subfolder)) and
                                      os.path.exists(os.path.join(dataset_dir, "labels", subfolder))
                           for subfolder in subfolders}

    # 如果有 train、val、test 子文件夹，分别生成对应的 jsonl 文件
    for subfolder, exists in subfolder_structure.items():
        if exists:
            image_dir = os.path.join(dataset_dir, "images", subfolder)
            label_dir = os.path.join(dataset_dir, "labels", subfolder)
            image_label_pairs = get_image_label_pairs(image_dir, label_dir)
            output_jsonl_path = f"{subfolder}.jsonl"  # 输出文件名，例如 train.jsonl, val.jsonl
            write_to_jsonl(image_label_pairs, classes, output_jsonl_path)
            print(f"Generated {output_jsonl_path} with {len(image_label_pairs)} pairs.")

    # 如果没有子文件夹，按照比例生成 train.jsonl 和 val.jsonl
    if not any(subfolder_structure.values()):
        image_dir = os.path.join(dataset_dir, "images")
        label_dir = os.path.join(dataset_dir, "labels")
        image_label_pairs = get_image_label_pairs(image_dir, label_dir)
        train_pairs, val_pairs = split_data(image_label_pairs, train_ratio=0.8)
        write_to_jsonl(train_pairs, classes, "train.jsonl")
        write_to_jsonl(val_pairs, classes, "val.jsonl")
        print(f"Generated train.jsonl with {len(train_pairs)} pairs.")
        print(f"Generated val.jsonl with {len(val_pairs)} pairs.")

