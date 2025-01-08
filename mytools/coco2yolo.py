import os
import json

def convert_coco_to_yolo_dataset(root_path, dataset_list):
    for dataset_name in dataset_list:
        dataset_path = os.path.join(root_path, dataset_name)
        images_dir = os.path.join(dataset_path, "images")
        labels_dir = os.path.join(dataset_path, "labels")
        coco_json = os.path.join(dataset_path, "annotations.json")
        
        # 创建labels文件夹
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        
        # 加载COCO格式的标注文件
        with open(coco_json, "r") as f:
            coco_data = json.load(f)

        # 获取类别映射
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        # 遍历每张图片的标注
        for img_info in coco_data["images"]:
            img_id = img_info["id"]
            img_name = img_info["file_name"]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # 获取该图片的所有标注
            annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == img_id]
            
            # 创建YOLO格式标签文件
            label_file = os.path.join(labels_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
            with open(label_file, "w") as label_f:
                for ann in annotations:
                    category_id = ann["category_id"] - 1  # YOLO的类别ID从0开始
                    bbox = ann["bbox"]

                    # 将COCO的bbox转换为YOLO格式
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height

                    # 写入标签
                    label_f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # 生成dataset.yaml文件
        yaml_file = os.path.join(dataset_path, "dataset.yaml")
        with open(yaml_file, "w") as yaml_f:
            yaml_f.write("names:\n")
            for i, name in enumerate(categories.values()):
                yaml_f.write(f"  {i}: {name}\n")

        print(f"转换完成: {dataset_name}")


# 使用方法
root_path = "/data1/lyf/formate_t/coco_format"  # 根路径
dataset_list = ["boat", "cheren", "jxsg", "laji", "smpfw","swim", "water", "yanhuo"]  # 数据集列表

convert_coco_to_yolo_dataset(root_path, dataset_list)
