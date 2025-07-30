import os
import torch
import shap
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from scripts.step3_train_deepfm_ms import DeepFM, DeepFMDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_path = os.path.join(project_root, "data", "amazon_all_beauty_interactions.csv")

df = pd.read_csv(csv_path)
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["user_id_enc"] = user_enc.fit_transform(df["user_id"])
df["item_id_enc"] = item_enc.fit_transform(df["item_id"])

dataset = DeepFMDataset(df)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
sample_batch = next(iter(dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(project_root, "models", "deepfm_model_ms.pt")
model = DeepFM(len(user_enc.classes_), len(item_enc.classes_)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device  # 自动获取模型所在设备

    def forward(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.long, device=self.device)  # 移到模型设备
        user_id, item_id = inputs[:, 0], inputs[:, 1]
        return self.model(user_id, item_id).detach().cpu().numpy()


X_shap_np = np.stack([
    sample_batch["user_id"].numpy(),
    sample_batch["item_id"].numpy()
], axis=1)

print("⏳ Running SHAP KernelExplainer...")
explainer = shap.KernelExplainer(Wrapper(model), X_shap_np[:100])
shap_values = explainer.shap_values(X_shap_np[:10])
print("📊 SHAP values (user_id, item_id):")
print(np.array(shap_values))

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
