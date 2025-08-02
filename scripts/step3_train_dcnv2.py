import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ========== Dataset ========== #
class DCNDataset(Dataset):
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

# ========== DCNv2 Enhanced Model ========== #
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

# ========== Training Script ========== #
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")

    df = pd.read_csv(csv_path)
    df.fillna("unknown", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1.0)
    df["price_norm"] = (df["price"] - df["price"].mean()) / (df["price"].std() + 1e-6)

    for col in ["user_id", "item_id", "category", "brand"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])

    dataset = DCNDataset(df)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    field_dims = {
        "user_id": df["user_id_enc"].nunique(),
        "item_id": df["item_id_enc"].nunique(),
        "category": df["category_enc"].nunique(),
        "brand": df["brand_enc"].nunique(),
        "price": 0
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNv2(field_dims).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    print("🚀 Start training DCNv2 (enhanced)...")
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
    model_path = os.path.join(model_dir, "dcnv2_model_full_ms.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ DCNv2 Enhanced Model saved to {model_path}")