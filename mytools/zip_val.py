import os
import json
import shutil
import zipfile

# 配置文件路径和目标文件夹
jsonl_file_path = "/data1/lyf/formate_t/coco_format/cheren/val.jsonl"  # 替换为你的 val.jsonl 文件路径
target_folder = "/data1/lyf/formate_t/coco_format/cheren"  # 目标文件夹
zip_file_name = "val.zip"  # 打包的压缩文件名

# 创建目标文件夹
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 读取 JSONL 文件并复制图像
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    for line in f:
        # 解析每一行 JSON
        data = json.loads(line.strip())
        image_path = data.get("image")  # 获取图像路径
        
        if image_path and os.path.exists(image_path):  # 确保图像路径存在
            # 复制图像到目标文件夹
            shutil.copy(image_path, target_folder)
        else:
            print(f"图像路径不存在: {image_path}")

# 打包目标文件夹为 ZIP 压缩文件
with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(target_folder):
        for file in files:
            file_path = os.path.join(root, file)
            # 将文件写入 ZIP 包
            zipf.write(file_path, os.path.relpath(file_path, target_folder))

print(f"图像已成功打包为 {zip_file_name}")
