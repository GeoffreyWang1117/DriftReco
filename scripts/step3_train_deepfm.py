import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    def __init__(self, dataframe):
        self.user_ids = torch.tensor(dataframe['user_id_enc'].values, dtype=torch.long)
        self.item_ids = torch.tensor(dataframe['item_id_enc'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }

class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, mlp_dims=[64, 32]):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        self.fm_bias = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], 1)
        )

    def forward(self, user_id, item_id):
        u = self.user_emb(user_id)
        v = self.item_emb(item_id)

        linear_part = (u + v).sum(dim=1)

        interaction = (u * v).sum(dim=1)

        deep_input = torch.cat([u, v], dim=1)
        deep_out = self.mlp(deep_input).squeeze()

        return self.fm_bias + linear_part + interaction + deep_out

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    df = pd.read_csv(os.path.join(project_root, "data", "amazon_all_beauty_interactions.csv"))

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["user_id_enc"] = user_enc.fit_transform(df["user_id"])
    df["item_id_enc"] = item_enc.fit_transform(df["item_id"])

    dataset = InteractionDataset(df)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(len(user_enc.classes_), len(item_enc.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("🚀 Start training DeepFM...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in dataloader:
            user_id = batch["user_id"].to(device)
            item_id = batch["item_id"].to(device)
            rating = batch["rating"].to(device)

            pred = model(user_id, item_id)
            loss = loss_fn(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rating)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(project_root, "models", "deepfm_model.pt"))
    print("✅ DeepFM 模型保存完毕")
