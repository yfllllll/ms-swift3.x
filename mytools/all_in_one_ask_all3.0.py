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


# 主程序入口
if __name__ == "__main__":
    # 用户选择推理引擎
    engine_type = 'pt'
    model_id_or_path = '/root/autodl-tmp/qwen7b-GPTQ-Int4'

    # 初始化推理引擎
    engine = initialize_engine(engine_type, model_id_or_path)
    request_config = RequestConfig(max_tokens=256, temperature=0)

    # 设置路径
    root_dir = "/ultralytics-main/datasets/VisDrone"
    folder_list = ["VisDrone2019-DET-test-dev"]
    saved_folder = 'pred_multi_engine'

    # 开始处理
    process_folders(root_dir, folder_list, saved_folder=saved_folder, engine=engine, request_config=request_config)

    print('Processing completed.')
