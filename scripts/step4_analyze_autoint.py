import os
import torch
import shap
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# ==== AutoInt 模型结构（与训练时一致）====
class AutoInt(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, num_heads=2):
        super().__init__()
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim) for name, dim in field_dims.items() if dim > 0
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

# ==== 路径设置 ====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")
model_path = os.path.join(project_root, "models", "autoint_model_full_ms.pt")

# ==== 加载数据 ====
df = pd.read_csv(csv_path)
df.fillna("unknown", inplace=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1.0)
df["price_norm"] = (df["price"] - df["price"].mean()) / (df["price"].std() + 1e-6)

encoders = {
    "user_id": LabelEncoder(),
    "item_id": LabelEncoder(),
    "category": LabelEncoder(),
    "brand": LabelEncoder()
}
for col in encoders:
    df[f"{col}_enc"] = encoders[col].fit_transform(df[col])

# ==== 初始化模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
field_dims = {
    "user_id": df["user_id_enc"].nunique(),
    "item_id": df["item_id_enc"].nunique(),
    "category": df["category_enc"].nunique(),
    "brand": df["brand_enc"].nunique(),
    "price": 0  # 连续值
}

model = AutoInt(field_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== SHAP 分析 ====
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

# ==== Fisher 信息 ====
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
        fisher = (param.grad ** 2).mean().item()
        print(f"{name}: {fisher:.6f}")