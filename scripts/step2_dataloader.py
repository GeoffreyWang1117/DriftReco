import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os

class FullBeautyDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()

        self.data['user_id_enc'] = self.user_encoder.fit_transform(self.data['user_id'])
        self.data['item_id_enc'] = self.item_encoder.fit_transform(self.data['item_id'])
        self.data['category_enc'] = self.category_encoder.fit_transform(self.data['category'].fillna("unknown"))
        self.data['brand_enc'] = self.brand_encoder.fit_transform(self.data['brand'].fillna("unknown"))
        self.data['price'] = self.data['price'].fillna(-1.0)

        self.user_ids = torch.tensor(self.data['user_id_enc'].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.data['item_id_enc'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.data['rating'].values, dtype=torch.float32)
        self.categories = torch.tensor(self.data['category_enc'].values, dtype=torch.long)
        self.brands = torch.tensor(self.data['brand_enc'].values, dtype=torch.long)
        self.prices = torch.tensor(self.data['price'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "category": self.categories[idx],
            "brand": self.brands[idx],
            "price": self.prices[idx]
        }

def get_full_dataloader(csv_path, batch_size=1024, shuffle=True, num_workers=0):
    dataset = FullBeautyDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset

if __name__ == "__main__":
    # 自动解析项目路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_file_path = os.path.join(project_root, "data", "amazon_all_beauty_full_clean.csv")

    dataloader, dataset = get_full_dataloader(csv_file_path)

    print(f"样本总数: {len(dataset)}")
    for batch in dataloader:
        print("user_id:", batch["user_id"][:5])
        print("item_id:", batch["item_id"][:5])
        print("category:", batch["category"][:5])
        print("brand:", batch["brand"][:5])
        print("price:", batch["price"][:5])
        break
