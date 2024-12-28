import os
import csv
from tensorboard.backend.event_processing import event_accumulator

# TensorBoard 日志文件路径
log_dir = "./ResNetWithSEAndConvRNN_epoch30"  # 替换为你的日志文件路径
output_csv = "ResNetWithSEAndConvRNN_epoch30.csv"  # 输出的CSV文件名

# 初始化 EventAccumulator 读取日志
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()  # 加载所有事件

# 获取所有可用的标记(tag)，排除 'Training Loss'
tags = ea.Tags().get('scalars', [])
excluded_tags = ["Training Loss"]  # 需要排除的标记
tags = [tag for tag in tags if tag not in excluded_tags]

# 创建一个 CSV 文件，将所有标记的数据存入其中
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入 CSV 文件的表头
    header = ['Step'] + tags  # 首列为 Step，后续列为每个标记的值
    writer.writerow(header)
    
    # 获取所有标记的最大步数
    max_steps = max([len(ea.Scalars(tag)) for tag in tags])

    # 整合数据并逐步写入
    for i in range(max_steps):
        row = [i]  # 每行以步数开头
        for tag in tags:
            scalars = ea.Scalars(tag)
            if i < len(scalars):
                row.append(scalars[i].value)  # 如果步数内有数据，写入值
            else:
                row.append(None)  # 如果没有数据，填充为 None
        writer.writerow(row)

print(f"所有数据（除去 {excluded_tags}）已保存到 {output_csv}")
