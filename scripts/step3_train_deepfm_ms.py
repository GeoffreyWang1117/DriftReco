import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

class DeepFMDataset(Dataset):
    def __init__(self, df):
        self.user_ids = torch.tensor(df['user_id_enc'].values, dtype=torch.long)
        self.item_ids = torch.tensor(df['item_id_enc'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }

class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fm_bias = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_id, item_id):
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)

        first_order = self.fm_bias
        interaction = (user_emb * item_emb).sum(dim=1, keepdim=True)
        x = torch.cat([user_emb, item_emb], dim=-1)
        deep_output = self.mlp(x)

        return (first_order + interaction + deep_output).squeeze(-1)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(project_root, "data", "amazon_all_beauty_interactions.csv")

    df = pd.read_csv(csv_path)
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df["user_id_enc"] = user_encoder.fit_transform(df["user_id"])
    df["item_id_enc"] = item_encoder.fit_transform(df["item_id"])

    dataset = DeepFMDataset(df)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(len(user_encoder.classes_), len(item_encoder.classes_)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("🚀 Start training...")
    for epoch in range(3):
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

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")

    model_path = os.path.join(project_root, "models", "deepfm_model_ms.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

