import os
import torch
import shap
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ========== DCNv2 增强结构定义 ==========
class DCNv2(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, num_cross_layers=6):
        super().__init__()
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in field_dims.items() if dim > 0
        })
        self.price_proj = nn.Linear(1, embedding_dim)
        self.input_dim = embedding_dim * 5

        self.cross_layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.input_dim) for _ in range(num_cross_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.input_dim) for _ in range(num_cross_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_cat, price):
        emb_list = [self.embedding_layers[field](x_cat[i]) for i, field in enumerate(self.embedding_layers)]
        price_embed = self.price_proj(price.unsqueeze(1))
        x0 = torch.cat(emb_list + [price_embed], dim=1)
        x = x0
        for layer, norm in zip(self.cross_layers, self.norm_layers):
            x = norm(x0 * layer(x) + x)
        return self.mlp(x).squeeze(1)

# ========== 路径与数据准备 ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")
model_path = os.path.join(project_root, "models", "dcnv2_model_full_ms.pt")

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

# ========== 模型加载 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
field_dims = {f: df[f"{f}_enc"].nunique() for f in ["user_id", "item_id", "category", "brand"]}
field_dims["price"] = 0

model = DCNv2(field_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========== SHAP 分析 ==========
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

print("📊 SHAP values shape:", np.array(shap_vals).shape)
for col, val in zip(columns, np.mean(shap_vals, axis=0)):
    print(f"{col}: {val:.6f}")

# ========== Fisher 信息估计 ==========
print("\n🔬 Estimating Fisher Information...")
sample = df.sample(n=128)
x_cat = [torch.tensor(sample[f"{f}_enc"].values, dtype=torch.long).to(device) for f in ["user_id", "item_id", "category", "brand"]]
price = torch.tensor(sample["price_norm"].values, dtype=torch.float32).to(device)
rating = torch.tensor(sample["rating"].values, dtype=torch.float32).to(device)

model.zero_grad()
pred = model(x_cat, price)
loss = nn.MSELoss()(pred, rating)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        fisher_val = (param.grad ** 2).mean().item()
        print(f"{name}: {fisher_val:.6f}")