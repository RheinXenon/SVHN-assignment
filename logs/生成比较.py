import pandas as pd
import os

# 文件路径列表
file_paths = [
    "CNN1_epoch30.csv",
    "CNN2_epoch30.csv",
    "FCN1_epoch30.csv",
    "MobileNetV2_se_epoch30.csv",
    "resnet_se_da_epoch30.csv",
    "resnet18_epoch30.csv",
    "resnet18_se_epoch30.csv",
    "ResNetWithSEAndConvRNN_epoch30.csv"
]

# 初始化结果字典
results = {
    "Model": [],
    "Accuracy": [],
    "F1_Score": []
}

# 遍历文件读取数据
for file_path in file_paths:
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取模型名称（从文件名中获取）
        model_name = os.path.basename(file_path).replace("_epoch30.csv", "")

        # 假设测试机上的准确度和F1分数列名为 'Test_Accuracy' 和 'Test_F1_Score'
        accuracy = df['Validation Accuracy'].iloc[-1]  # 获取最后一行的值
        f1_score = df['Validation F1 Score'].iloc[-1]

        # 添加到结果字典中
        results["Model"].append(model_name)
        results["Accuracy"].append(accuracy)
        results["F1_Score"].append(f1_score)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# 将结果转换为数据框
results_df = pd.DataFrame(results)

# 保存为CSV文件
output_path = "model_performance_comparison.csv"
results_df.to_csv(output_path, index=False)

# 打印结果
print("Model Performance Comparison Table:")
print(results_df)

print(f"The comparison table has been saved to {output_path}")
