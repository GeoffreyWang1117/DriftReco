import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class AutoIntDataset(Dataset):
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
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'category': self.category_ids[idx],
            'brand': self.brand_ids[idx],
            'price': self.prices[idx],
            'rating': self.ratings[idx]
        }

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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")

    df = pd.read_csv(csv_path)
    df.fillna("unknown", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1.0)
    df["price_norm"] = (df["price"] - df["price"].mean()) / (df["price"].std() + 1e-6)

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    cat_enc = LabelEncoder()
    brand_enc = LabelEncoder()

    df["user_id_enc"] = user_enc.fit_transform(df["user_id"])
    df["item_id_enc"] = item_enc.fit_transform(df["item_id"])
    df["category_enc"] = cat_enc.fit_transform(df["category"])
    df["brand_enc"] = brand_enc.fit_transform(df["brand"])

    dataset = AutoIntDataset(df)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    field_dims = {
        "user_id": df["user_id_enc"].nunique(),
        "item_id": df["item_id_enc"].nunique(),
        "category": df["category_enc"].nunique(),
        "brand": df["brand_enc"].nunique(),
        "price": 0
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoInt(field_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("🚀 Start training AutoInt...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x_cat = [batch['user_id'].to(device),
                     batch['item_id'].to(device),
                     batch['category'].to(device),
                     batch['brand'].to(device)]
            price = batch['price'].to(device)
            rating = batch['rating'].to(device)

            pred = model(x_cat, price)
            loss = loss_fn(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rating)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}")

    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "autoint_model_full_ms.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ AutoInt Model saved to {model_path}")