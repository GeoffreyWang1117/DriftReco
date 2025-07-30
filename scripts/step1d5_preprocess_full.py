import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# 初始化 BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert.to(device)

# 路径设定
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(project_root, "data")

# 输入文件（step1生成的交互数据）
interact_path = os.path.join(data_dir, "amazon_all_beauty_interactions.csv")
df_interact = pd.read_csv(interact_path)

# 原始Parquet路径
parquet_path = os.path.join(data_dir, "full-00000-of-00001.parquet")
df_raw = pd.read_parquet(parquet_path)

# 筛选并重命名关键字段
df_raw = df_raw.rename(columns={
    "parent_asin": "item_id",
    "average_rating": "rating"
})
df_raw = df_raw[["item_id", "review_text", "product_title", "brand", "category", "price"]]

# 合并交互数据与原始信息（左连接，保留交互）
df = pd.merge(df_interact, df_raw, how="left", on="item_id")

# 生成文本输入字段
df["text_input"] = (
    df["product_title"].fillna("") + " " +
    df["review_text"].fillna("")
).str.strip()

# BERT嵌入
def encode_bert_batch(texts, max_len=64):
    inputs = tokenizer(texts, padding="max_length", truncation=True,
                       max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu()  # CLS token

print("🚀 Encoding BERT embeddings...")
batch_size = 64
text_embeds = []
for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df["text_input"].iloc[i:i + batch_size].tolist()
    batch_embed = encode_bert_batch(batch_texts)
    text_embeds.append(batch_embed)

# 合并结果
text_embeds = torch.cat(text_embeds, dim=0)
print(f"✅ Text Embeddings shape: {text_embeds.shape}")

# 添加 768 列文本特征
for i in range(768):
    df[f"text_feat_{i}"] = text_embeds[:, i].numpy()

# 最终输出字段
final_columns = (
    ["user_id", "item_id", "rating", "timestamp"] +
    [f"text_feat_{i}" for i in range(768)]
)
df_final = df[final_columns]

# 输出到 enriched 文件
output_path = os.path.join(data_dir, "amazon_all_beauty_enriched.csv")
df_final.to_csv(output_path, index=False)
print(f"📁 Saved enriched CSV to: {output_path}")
print(f"📊 Final samples: {len(df_final)}")
