# 文件结构
# datasets/test/test/1.png 测试集的示例图片位置
# datasets/train/train/1.png 测试集的示例图片位置
# datasets/test_digitStruct.json 测试集标记json
# datasets/train_digitStruct.json 训练集标记json
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

# 解析 JSON 文件
def parse_digit_struct_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    parsed_data = []
    for item in data:
        parsed_data.append({
            'filename': item['filename'],
            'bbox': item['bbox']
        })
    return parsed_data

# 显示前32张图片和它们的标签
def display_images_with_labels(data, images_dir, num_images=32):
    plt.figure(figsize=(12, 12))
    
    for i, item in enumerate(data[:num_images]):
        img_path = os.path.join(images_dir, item['filename'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # 打开图片
        img = Image.open(img_path)
        plt.subplot(4, 8, i + 1)  # 创建 4x8 的网格
        plt.imshow(img)
        plt.axis('off')
        
        # 显示标签
        labels = item['bbox']['label']
        label_text = ''.join([str(int(l)) for l in labels])  # 将标签拼接成字符串
        plt.title(label_text, fontsize=8)
    
    plt.tight_layout()
    plt.show()

# 文件路径
train_json_file = './datasets/train_digitStruct.json'
train_images_dir = './datasets/train/train'

# 加载数据
train_data = parse_digit_struct_json(train_json_file)

# 显示前32张图片和标签
display_images_with_labels(train_data, train_images_dir)
