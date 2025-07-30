import pandas as pd
import os
import json

# 设置数据来源（Hugging Face Parquet）
parquet_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw_meta_All_Beauty/full-00000-of-00001.parquet"

# 获取项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

print("🚀 正在从 Hugging Face 加载原始 Parquet 文件...")
df = pd.read_parquet(parquet_url)
print("✅ 加载完成")

# 字段预览
print("📋 字段名:", df.columns.tolist())
print("📌 第一条记录示例:\n", df.iloc[0].to_dict())

# 仅保留重要字段 + 清洗处理
fields_to_keep = [
    "parent_asin",          # 作为 item_id
    "title",                # 商品标题
    "main_category",        # 所属大类
    "average_rating",       # 平均评分
    "rating_number",        # 评分数
    "store",                # 商家名称
    "price",                # 价格，字符串或 None
    "details",              # JSON 字符串，保留
]

# 新构造 user_id + timestamp
df_clean = df[fields_to_keep].copy()
df_clean["user_id"] = [f"user_{i % 10000}" for i in range(len(df_clean))]
df_clean["timestamp"] = pd.Timestamp.now().timestamp()

# 处理字段命名
df_clean.rename(columns={
    "parent_asin": "item_id",
    "title": "product_title",
    "main_category": "category",
    "average_rating": "rating",
    "store": "brand"
}, inplace=True)

# 将价格转换为 float，非法值置为 -1.0
def safe_parse_price(x):
    try:
        return float(x)
    except:
        return -1.0

df_clean["price"] = df_clean["price"].apply(safe_parse_price)

# 丢弃无 item_id 或 rating 的行
df_clean.dropna(subset=["item_id", "rating"], inplace=True)

# 保存为 CSV
csv_path = os.path.join(data_dir, "amazon_all_beauty_full_clean.csv")
df_clean.to_csv(csv_path, index=False)

print(f"📦 清洗后 CSV 文件保存至: {csv_path}")
print(f"🔢 样本总数: {len(df_clean)}")
