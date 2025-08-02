import os
import torch
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from step3_train_dlrm import DLRM  # 确保模型结构可复用

# ========== Dataset ========== #
class DLRMDataset(Dataset):
    def __init__(self, df):
        self.user_ids = torch.tensor(df['user_id_enc'].values, dtype=torch.long)
        self.item_ids = torch.tensor(df['item_id_enc'].values, dtype=torch.long)
        self.category_ids = torch.tensor(df['category_enc'].values, dtype=torch.long)
        self.brand_ids = torch.tensor(df['brand_enc'].values, dtype=torch.long)
        self.prices = torch.tensor(df['price_norm'].values, dtype=torch.float32)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            [self.user_ids[idx], self.item_ids[idx], self.category_ids[idx], self.brand_ids[idx]],
            self.prices[idx],
            self.ratings[idx]
        )

# ========== Load Model & Data ========== #
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")
model_path = os.path.join(project_root, "models", "dlrm_model_full_ms.pt")

df = pd.read_csv(csv_path)
df.fillna("unknown", inplace=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1.0)
df["price_norm"] = (df["price"] - df["price"].mean()) / (df["price"].std() + 1e-6)

for col in ["user_id", "item_id", "category", "brand"]:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col])

dataset = DLRMDataset(df)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

field_dims = {
    "user_id": df["user_id_enc"].nunique(),
    "item_id": df["item_id_enc"].nunique(),
    "category": df["category_enc"].nunique(),
    "brand": df["brand_enc"].nunique(),
    "price": 0
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DLRM(field_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========== SHAP Analysis ========== #
print("🔍 Running SHAP...")

batch = next(iter(dataloader))
x_cat_batch = [x.to(device) for x in batch[0]]
price_batch = batch[1].to(device)

def model_forward(x_combined):
    # x_combined: shape (batch_size, 5), last is price
    x_cat_tensors = [x_combined[:, i].long().to(device) for i in range(4)]
    price_tensor = x_combined[:, 4].float().to(device)
    return model(x_cat_tensors, price_tensor).detach().cpu().numpy()

# 将 Tensor 转为 numpy
input_data = torch.stack(batch[0] + [batch[1]], dim=1).float()
input_data_np = input_data.cpu().numpy()
# 修复后的模型 forward 包装器
def model_forward(x_combined_np):
    x_combined = torch.tensor(x_combined_np, dtype=torch.float32).to(device)
    x_cat_tensors = [x_combined[:, i].long() for i in range(4)]  # user_id/item_id/category/brand
    price_tensor = x_combined[:, 4]
    with torch.no_grad():
        output = model(x_cat_tensors, price_tensor)
    return output.cpu().numpy()

explainer = shap.Explainer(model_forward, input_data_np)  # 或 masker=shap.maskers.Independent(...)
shap_values = explainer(input_data_np[:5])

feature_names = ["user_id", "item_id", "category", "brand", "price_norm"]
shap_vals = np.abs(shap_values.values).mean(axis=0)

for name, val in zip(feature_names, shap_vals):
    print(f"{name}: {val:.6f}")

plt.bar(feature_names, shap_vals)
plt.title("SHAP Feature Importances (DLRM)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(project_root, "outputs", "shap_dlrm_barplot.png"))
print("📊 SHAP barplot saved.")

# ========== Fisher Information ========== #
print("🔬 Estimating Fisher Information...")
fisher_info = {}
model.zero_grad()
for batch in dataloader:
    x_cat = [x.to(device) for x in batch[0]]
    price = batch[1].to(device)
    target = batch[2].to(device)
    pred = model(x_cat, price)
    loss = nn.MSELoss()(pred, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher_info[name] = param.grad.detach().pow(2).mean().item()

    break  # 仅估计一批次

for name, val in fisher_info.items():
    print(f"{name}: {val:.6f}")

# ========== Save for step4e ========== #
os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
np.save(os.path.join(project_root, "outputs", "shap_dlrm.npy"), shap_vals)
np.save(os.path.join(project_root, "outputs", "fisher_dlrm.npy"), fisher_info)