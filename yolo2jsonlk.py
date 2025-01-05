import os
import yaml
import json

def get_classes_from_yaml(yaml_path):
    """
    从dataset.yaml文件中解析出类别名列表
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    classes = [v for k, v in data.get('names', {}).items()]
    return classes

def get_image_label_pairs(data_dir):
    """
    获取数据目录下图像和标注文件的路径对
    这里假设图像在images文件夹，标注在labels文件夹，且文件名（除后缀外）一一对应
    """
    image_extensions = ('.jpg', '.png', '.jpeg')  # 支持的图像文件扩展名
    label_extension = '.txt'
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")
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

if __name__ == "__main__":
    dataset_dir = "/data1/lyf/datasets/VisDrone/VisDrone2019-DET-test-dev"  # 替换为你的数据集实际目录路径
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    output_jsonl_path = "output.jsonl"  # 输出的jsonl文件路径，可按需修改
    classes = get_classes_from_yaml(yaml_path)
    image_label_pairs = get_image_label_pairs(dataset_dir)
    write_to_jsonl(image_label_pairs, classes, output_jsonl_path)
    
