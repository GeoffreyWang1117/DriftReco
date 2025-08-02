import os
import pandas as pd
import matplotlib.pyplot as plt

# ==== 路径设置 ====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

outputs_dir = os.path.join(project_root, "outputs")
compare_dir = os.path.join(outputs_dir, "comparison")
os.makedirs(compare_dir, exist_ok=True)

# ==== 自动发现所有模型目录 ====
model_dirs = [
    name for name in os.listdir(outputs_dir)
    if os.path.isdir(os.path.join(outputs_dir, name))
       and name != "comparison"
]

# ==== 收集 SHAP 平均值 ====
shap_means = {}

for model in model_dirs:
    shap_path = os.path.join(outputs_dir, model, "shap_values.csv")
    if os.path.exists(shap_path):
        df = pd.read_csv(shap_path)
        shap_means[model] = df.mean()

# 合并 SHAP 平均值表格
shap_compare_df = pd.DataFrame(shap_means)

# ==== 绘图：SHAP ====
if not shap_compare_df.empty:
    plt.figure(figsize=(10, 5))
    shap_compare_df.plot(kind="barh", ax=plt.gca())
    plt.title("Average SHAP Value per Feature (All Models)")
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "shap_comparison_barplot.png"))
    plt.close()
    shap_compare_df.to_csv(os.path.join(compare_dir, "shap_comparison.csv"))
    print("✅ SHAP comparison saved.")
else:
    print("⚠️ No SHAP data found in any model directory.")

# ==== 合并 Fisher 信息 ====
fisher_dfs = []

for model in model_dirs:
    fisher_path = os.path.join(outputs_dir, model, "fisher_info.csv")
    if os.path.exists(fisher_path):
        df = pd.read_csv(fisher_path)
        df = df.rename(columns={"FisherInformation": f"Fisher_{model}"})
        fisher_dfs.append(df)

# 逐个合并
fisher_compare_df = None
for df in fisher_dfs:
    if fisher_compare_df is None:
        fisher_compare_df = df
    else:
        fisher_compare_df = pd.merge(fisher_compare_df, df, on="Parameter", how="outer")

# ==== 绘图：Fisher ====
if fisher_compare_df is not None:
    fisher_compare_df.fillna(0, inplace=True)
    fisher_compare_df.to_csv(os.path.join(compare_dir, "fisher_comparison.csv"), index=False)

    # 提取所有模型中最大值作为排序依据
    fisher_compare_df["MaxValue"] = fisher_compare_df.drop("Parameter", axis=1).max(axis=1)
    top_params = fisher_compare_df.sort_values(by="MaxValue", ascending=False).head(10)
    top_params.drop("MaxValue", axis=1).set_index("Parameter").plot(kind="barh", figsize=(10, 6))
    plt.title("Top 10 Parameters by Fisher Information (All Models)")
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "fisher_comparison_barplot.png"))
    plt.close()
    print("✅ Fisher comparison saved.")
else:
    print("⚠️ No Fisher data found in any model directory.")