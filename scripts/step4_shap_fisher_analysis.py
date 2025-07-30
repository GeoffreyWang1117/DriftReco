import os
import torch
import shap
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from scripts.step3_train_mlp import RecSysModel
from scripts.step2_dataloader import get_dataloader

# ✅ 模型与设备设置
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
model_path = os.path.join(project_root, "models", "mlp_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载数据
dataloader, dataset = get_dataloader(shuffle=False)
sample_batch = next(iter(dataloader))

# ✅ 定义 SHAP 包装器（必须输出 numpy）
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_numpy):
        inputs = torch.tensor(inputs_numpy, dtype=torch.long, device=device)
        user_id, item_id = inputs[:, 0], inputs[:, 1]
        with torch.no_grad():
            output = self.model(user_id, item_id).detach().cpu().numpy()
        return output

# ✅ 构建模型并加载参数
model = RecSysModel(
    num_users=int(dataset.user_ids.max().item()) + 1,
    num_items=int(dataset.item_ids.max().item()) + 1
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

wrapped_model = Wrapper(model)

# ✅ 生成 SHAP 输入（转换为 numpy）
X_shap = torch.stack([sample_batch['user_id'], sample_batch['item_id']], dim=1).cpu().numpy()

# ✅ 使用 KernelExplainer（对黑盒模型兼容性强）
print("⏳ Running SHAP KernelExplainer...")
explainer = shap.KernelExplainer(wrapped_model, X_shap[:100])  # background
shap_values = explainer.shap_values(X_shap[:10])  # explain 10 samples

# ✅ 输出 SHAP 值
print("📊 SHAP values (user_id, item_id):")
print(np.array(shap_values))

# ✅ Fisher 信息估计（基于 MSE 梯度平方）
def estimate_fisher(model, dataloader):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)

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

# ✅ 计算 Fisher 信息
print("⏳ Estimating Fisher Information...")
fisher_info = estimate_fisher(model, dataloader)

# ✅ 输出 Fisher 信息摘要
print("📈 Estimated Fisher Information (squared gradients):")
for name, value in fisher_info.items():
    print(f"{name}: mean={value.mean().item():.4f}, std={value.std().item():.4f}")