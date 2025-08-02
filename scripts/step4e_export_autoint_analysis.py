import os
import torch
import shap
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============== 模型定义 ==============
class AutoInt(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, num_heads=2):
        super().__init__()
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in field_dims.items() if dim > 0
        })
        self.price_proj = nn.Linear(1, embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_cat, price):
        embed_list = [self.embedding_layers[field](x_cat[i]) for i, field in enumerate(self.embedding_layers)]
        price_embed = self.price_proj(price.unsqueeze(1)).unsqueeze(1)
        inputs = torch.stack(embed_list + [price_embed.squeeze(1)], dim=1)
        attn_output, _ = self.attention(inputs, inputs, inputs)
        pooled = attn_output.mean(dim=1)
        return self.mlp(pooled).squeeze(1)

# ============== 路径与数据准备 ==============
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")
model_path = os.path.join(project_root, "models", "autoint_model_full_ms.pt")
export_dir = os.path.join(project_root, "outputs", "autoint")
os.makedirs(export_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df.fillna("unknown", inplace=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1.0)
df["price_norm"] = (df["price"] - df["price"].mean()) / df["price"].std()

encoders = {
    "user_id": LabelEncoder(),
    "item_id": LabelEncoder(),
    "category": LabelEncoder(),
    "brand": LabelEncoder()
}
for col in encoders:
    df[f"{col}_enc"] = encoders[col].fit_transform(df[col])

# ============== 加载模型 ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
field_dims = {f: df[f"{f}_enc"].nunique() for f in ["user_id", "item_id", "category", "brand"]}
field_dims["price"] = 0  # 连续值字段

model = AutoInt(field_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ============== SHAP 分析 ==============
class WrapperModel:
    def __call__(self, X_numpy):
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(device)
        x_cat = [X_tensor[:, i].long() for i in range(4)]
        price = X_tensor[:, 4]
        with torch.no_grad():
            return model(x_cat, price).cpu().numpy()

print("🔍 Running SHAP...")
X_np = df[["user_id_enc", "item_id_enc", "category_enc", "brand_enc", "price_norm"]].values[:20]
explainer = shap.KernelExplainer(WrapperModel(), X_np[:10])
shap_vals = explainer.shap_values(X_np[:5])
columns = ["user_id", "item_id", "category", "brand", "price_norm"]

# 保存 SHAP 为 .npy 和 CSV
np.save(os.path.join(export_dir, "shap_values.npy"), shap_vals)
shap_df = pd.DataFrame(shap_vals, columns=columns)
shap_df.to_csv(os.path.join(export_dir, "shap_values.csv"), index=False)

# 柱状图可视化
mean_shap = shap_df.mean()
plt.figure(figsize=(8, 4))
mean_shap.plot(kind="barh", color="skyblue")
plt.title("Average SHAP Value per Feature (AutoInt)")
plt.tight_layout()
plt.savefig(os.path.join(export_dir, "shap_barplot.png"))
plt.close()
print("📊 SHAP barplot saved.")

# ============== Fisher 信息 ==============
print("🔬 Estimating Fisher Information...")
sample = df.sample(n=128)
x_cat = [torch.tensor(sample[f"{f}_enc"].values, dtype=torch.long).to(device) for f in ["user_id", "item_id", "category", "brand"]]
price = torch.tensor(sample["price_norm"].values, dtype=torch.float32).to(device)
rating = torch.tensor(sample["rating"].values, dtype=torch.float32).to(device)

model.zero_grad()
pred = model(x_cat, price)
loss = nn.MSELoss()(pred, rating)
loss.backward()

fisher_data = []
for name, param in model.named_parameters():
    if param.grad is not None:
        fisher_val = (param.grad ** 2).mean().item()
        fisher_data.append((name, fisher_val))
        print(f"{name}: {fisher_val:.6f}")

fisher_df = pd.DataFrame(fisher_data, columns=["Parameter", "FisherInformation"])
fisher_df.to_csv(os.path.join(export_dir, "fisher_info.csv"), index=False)
np.save(os.path.join(export_dir, "fisher_info.npy"), np.array(fisher_data, dtype=object))
print("✅ SHAP, Fisher and visual outputs saved.")