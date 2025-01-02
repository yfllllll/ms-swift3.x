import json
import os
import random
import re
from tqdm import tqdm  # 添加进度条库
from collections import defaultdict

def restore_bbox(bboxes, height, width, BOX_SCALE=999):
    """
    恢复边界框的尺寸，并归一化到1000以内。
    """
    restored_bboxes = []
    for bbox in bboxes:
        bbox = [
            int(bbox[0] / BOX_SCALE * max(height, width)),
            int(bbox[1] / BOX_SCALE * max(height, width)),
            int(bbox[2] / BOX_SCALE * max(height, width)),
            int(bbox[3] / BOX_SCALE * max(height, width)),
        ]
        if height != width:
            delta = (width - height) // 2 if height < width else (height - width) // 2
            if height < width:
                bbox[1] -= delta
                bbox[3] -= delta
            else:
                bbox[0] -= delta
                bbox[2] -= delta
                
        bbox = [max(0, coord) for coord in bbox]
        #归一化到[0, 999]
        # bbox = [int(coord / max(height, width) * 1000) for coord in bbox]
        bbox = [int(coord / dim * 999) for coord, dim in zip(bbox, [width, height, width, height])]
        restored_bboxes.append(bbox)
    return restored_bboxes

def process_bbox_in_query_or_response(text, height, width, resset_bbox=True):
    """
    检查并处理文本中的矩形框，恢复其尺寸并归一化。
    """
    if text.startswith('<image>\n'):
        text = text.replace('<image>\n', '<image>', 1)
    if '<box>' in text:
        bbox_match = re.findall(r'<box>\[(.*?)\]</box>', text)
        if bbox_match:
            bbox_data = json.loads(f"[{bbox_match[0]}]")
            if resset_bbox:
                bbox_data = restore_bbox(bbox_data, height, width)
            bbox_data = json.dumps(bbox_data)
            text = re.sub(r'<box>.*?</box>', f'<box_start>{bbox_data}<box_end>', text)
    return text

def process_grounding_response(response, height, width):
    """
    处理 grounding 任务的 response，支持多目标和关系解析。
    """
    objects = defaultdict(list)
    formatted_response = ""

    # 匹配 <ref> 和 <box> 配对的情况
    if('<ref>'in response):
        matches = re.findall(r'<ref>(.*?)</ref>.*?<box_start>\[(.*?)\]<box_end>', response)
        for classname, bbox_str in matches:
            bbox_data = json.loads(f"[{bbox_str}]")
            restored_bboxes = restore_bbox(bbox_data, height, width)
            objects[classname].extend(restored_bboxes)
    else:   
        # 匹配只有 <box> 的情况
        box_matches = re.findall(r'<box_start>\[(.*?)\]<box_end>', response)
        if box_matches:
            for bbox_str in box_matches:
                bbox_data = json.loads(f"[{bbox_str}]")
                restored_bboxes = restore_bbox(bbox_data, height, width)
                objects["未知"].extend(restored_bboxes)  # 将类别设置为 "未知"

    # 格式化结果
    total_objects = sum(len(b) for b in objects.values())
    formatted_response += f"本张影像存在{total_objects}个目标:"
    for classname, bboxes in objects.items():
        for bbox in bboxes:
            if classname == "未知":
                formatted_response += f"<|box_start|>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})<|box_end|>"
            else:
                formatted_response += f"<|object_ref_start|>{classname}<|object_ref_end|><|box_start|>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})<|box_end|>"

    return formatted_response.strip()


def valid_grounding(json_data):
    try:
        question = json_data['conversations'][0]['value']
        answer = json_data['conversations'][1]['value']

        # 检查错误的回答：包含 'None' 或 '不存在此类别'
        if 'None' in answer or '不存在此类别' in answer:
            print(f"回答错误")
            return False
        
        # 正则表达式(回答标签)
        pattern = r'<ref>(.*?)<\/ref>'
        matches = re.findall(pattern, answer)
        for i in matches:
            if i not in question:
                print(f"输入目标与输出目标不匹配")
                return False
        return True
    
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return False


def process_data(data, img_dir_labels, images_dir, dataset_name):
    processed_data = []
    height, width = [data["height"], data["width"]]
    image_path = os.path.join(images_dir, data["image"])
    system_message = {"role": "system", "content": "你是个有用无害的助手"}
    messages = [system_message]

    for conversation in data["conversations"]:
        if conversation["from"] == "human":
            query = conversation["value"]
            query = process_bbox_in_query_or_response(query, height, width, resset_bbox=True)
            messages.append({"role": "user", "content": query})

        elif conversation["from"] == "gpt":
            response = conversation["value"]
            response = process_bbox_in_query_or_response(response, height, width, resset_bbox=False)
            if '<box_start>' in response:  # 判断是否是 grounding 任务
                if not valid_grounding(data):
                    return []
                response = process_grounding_response(response, height, width)
            messages.append({"role": "assistant", "content": response})

    processed_data.append({
        "messages": messages,
        "images": [image_path]
    })

    return processed_data

def save_data_as_jsonl(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保输出目录存在
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')

def process_dataset_v3(img_dir_labels, output_dir, split_ratio=0.8):
    for dataset_name, dataset_info in img_dir_labels.items():
        img_dir = dataset_info['images']
        label_dir = dataset_info['annotations']

        all_data = []

        label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]

        for label_file in tqdm(label_files, desc=f"Processing dataset: {dataset_name}"):
            label_path = os.path.join(label_dir, label_file)
            try:
                with open(label_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    converted_data = process_data(data, img_dir_labels, img_dir, dataset_name)
                    all_data.extend(converted_data)
            except Exception as e:
                print(f"处理文件 {label_path} 时发生错误: {e}")

        # 打乱数据顺序并划分为训练集和验证集
        random.shuffle(all_data)
        train_size = int(len(all_data) * split_ratio)
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]

        # 保存训练集
        train_output_path = os.path.join(output_dir, f'{dataset_name}_train.jsonl')
        save_data_as_jsonl(train_data, train_output_path)
        print(f"已保存训练集数据到 {train_output_path}")

        # 保存验证集
        val_output_path = os.path.join(output_dir, f'{dataset_name}_val.jsonl')
        save_data_as_jsonl(val_data, val_output_path)
        print(f"已保存验证集数据到 {val_output_path}")


# 测试数据列表
img_dir_labels = {
    'Tower_bigdata':{
        'images':"/ms-swift/dataset/Tower_bigdata/images",
        'annotations':"/ms-swift/dataset/Tower_bigdata/annotations/",
    },
    "Tower_data_1":{
        "images":"/ms-swift/dataset/Tower_data_1/images/",
        "annotations":"/ms-swift/dataset/Tower_data_1/annotations/"
    }
   }


output_dir = '/ms-swift/dataset/fromVary_3.0'  # 修改为实际输出路径
process_dataset_v3(img_dir_labels, output_dir, split_ratio=1.0)
