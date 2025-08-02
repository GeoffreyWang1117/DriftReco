import os
import torch
import shap
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DeepFMFull(nn.Module):
    def __init__(self, field_dims, embedding_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim) for name, dim in field_dims.items() if dim > 0
        })
        self.fm_bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * (len(field_dims) - 1), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_cat, price):
        emb_list = [self.embeddings[field](x_cat[i]) for i, field in enumerate(self.embeddings)]
        fm_interaction = sum([
            e1 * e2 for i, e1 in enumerate(emb_list) for j, e2 in enumerate(emb_list) if i < j
        ])
        x_mlp = torch.cat(emb_list, dim=-1)
        deep_output = self.mlp(x_mlp)
        return (self.fm_bias + fm_interaction.sum(dim=1, keepdim=True) + deep_output + price.unsqueeze(1)).squeeze(1)

# ==== 路径设置 ====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")
model_path = os.path.join(project_root, "models", "deepfm_model_full_ms.pt")

# ==== 加载并预处理数据 ====
df = pd.read_csv(csv_path)
df.fillna("unknown", inplace=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1.0)

# ==== price 归一化 ====
valid_prices = df["price"].replace(-1.0, np.nan)
min_price, max_price = valid_prices.min(), valid_prices.max()
df["price_norm"] = df["price"].apply(lambda x: (x - min_price) / (max_price - min_price) if x != -1.0 else -1.0)

# ==== 编码分类变量 ====
user_enc = LabelEncoder()
item_enc = LabelEncoder()
cat_enc = LabelEncoder()
brand_enc = LabelEncoder()

df["user_id_enc"] = user_enc.fit_transform(df["user_id"])
df["item_id_enc"] = item_enc.fit_transform(df["item_id"])
df["category_enc"] = cat_enc.fit_transform(df["category"])
df["brand_enc"] = brand_enc.fit_transform(df["brand"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载模型结构与权重 ====
field_dims = {
    "user_id": df["user_id_enc"].nunique(),
    "item_id": df["item_id_enc"].nunique(),
    "category": df["category_enc"].nunique(),
    "brand": df["brand_enc"].nunique(),
    "price": 0  # continuous, not embedded
}
model = DeepFMFull(field_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== SHAP 包装器 ====
class WrapperModel:
    def __call__(self, X_numpy):
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(device)
        x_cat = [X_tensor[:, i].long() for i in range(4)]
        price = X_tensor[:, 4]
        with torch.no_grad():
            return model(x_cat, price).cpu().numpy()

print("🔎 Running SHAP on small batch...")
X_np = df[["user_id_enc", "item_id_enc", "category_enc", "brand_enc", "price_norm"]].values[:20]
explainer = shap.KernelExplainer(WrapperModel(), X_np[:10])
shap_vals = explainer.shap_values(X_np[:5])
print("📊 SHAP values shape:", shap_vals.shape)

columns = ["user_id", "item_id", "category", "brand", "price_norm"]
print("🧮 Feature importance (avg over examples):")
for i, col in enumerate(columns):
    print(f"{col}: {shap_vals[:, i].mean():.6f}")

# ==== Fisher 信息估计 ====
print("\n🔬 Fisher Information:")
sample = df.sample(n=128)
x_cat = [
    torch.tensor(sample["user_id_enc"].values, dtype=torch.long).to(device),
    torch.tensor(sample["item_id_enc"].values, dtype=torch.long).to(device),
    torch.tensor(sample["category_enc"].values, dtype=torch.long).to(device),
    torch.tensor(sample["brand_enc"].values, dtype=torch.long).to(device),
]
price = torch.tensor(sample["price_norm"].values, dtype=torch.float32).to(device)
rating = torch.tensor(sample["rating"].values, dtype=torch.float32).to(device)

model.zero_grad()
pred = model(x_cat, price)
loss = nn.MSELoss()(pred, rating)
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        fi = (param.grad ** 2).mean().item()
        print(f"{name}: Fisher Information = {fi:.6f}")