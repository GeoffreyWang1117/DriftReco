import os
import torch
import shap
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from scripts.step3_train_deepfm import DeepFM, InteractionDataset

# 数据路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(project_root, "data", "amazon_all_beauty_interactions.csv")

# 加载数据并编码
df = pd.read_csv(csv_path)
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["user_id_enc"] = user_enc.fit_transform(df["user_id"])
df["item_id_enc"] = item_enc.fit_transform(df["item_id"])

dataset = InteractionDataset(df)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(project_root, "models", "deepfm_model.pt")
model = DeepFM(len(user_enc.classes_), len(item_enc.classes_)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# SHAP 模型封装
# SHAP 模型封装
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        # ✅ 修复：numpy -> Tensor，才能使用 .long()
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        user_id = inputs_tensor[:, 0].long().to(device)
        item_id = inputs_tensor[:, 1].long().to(device)
        with torch.no_grad():
            return self.model(user_id, item_id).cpu().numpy()

wrapped_model = Wrapper(model)

# 准备 SHAP 输入（转换为 numpy）
sample_batch = next(iter(dataloader))
X_shap = torch.stack([sample_batch['user_id'], sample_batch['item_id']], dim=1).float()
X_shap_np = X_shap.numpy()  # ✅ 修复点

# SHAP 值计算
print("⏳ Running SHAP KernelExplainer...")
explainer = shap.KernelExplainer(wrapped_model, X_shap_np[:100])
shap_values = explainer.shap_values(X_shap_np[:10])
print("📊 SHAP values (user_id, item_id):")
print(np.array(shap_values))

# Fisher 信息估计
def estimate_fisher(model, dataloader):
    model.eval()
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for batch in dataloader:
        model.zero_grad()
        user_id = batch["user_id"].to(device)
        item_id = batch["item_id"].to(device)
        rating = batch["rating"].to(device)

        pred = model(user_id, item_id)
        loss = torch.nn.functional.mse_loss(pred, rating)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

    for name in fisher:
        fisher[name] /= len(dataloader)

    return fisher

print("⏳ Estimating Fisher Information...")
fisher_info = estimate_fisher(model, dataloader)
for name, value in fisher_info.items():
    print(f"{name}: mean={value.mean().item():.4f}, std={value.std().item():.4f}")
