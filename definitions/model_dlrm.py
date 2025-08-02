import torch
import torch.nn as nn

class DLRMModel(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, bottom_mlp_sizes=[64, 32], top_mlp_sizes=[64, 32, 1]):
        super().__init__()
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in field_dims.items() if dim > 0
        })
        self.price_proj = nn.Linear(1, embedding_dim)

        # Bottom MLP (dense feature)
        self.bottom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, bottom_mlp_sizes[0]),
            nn.ReLU(),
            nn.Linear(bottom_mlp_sizes[0], bottom_mlp_sizes[1]),
            nn.ReLU()
        )

        num_features = len(self.embedding_layers) + 1  # +1 for price
        num_interactions = num_features * (num_features - 1) // 2
        top_input_dim = num_interactions + bottom_mlp_sizes[-1]

        top_layers = []
        for i in range(len(top_mlp_sizes) - 1):
            top_layers.append(nn.Linear(top_input_dim if i == 0 else top_mlp_sizes[i], top_mlp_sizes[i + 1]))
            if i < len(top_mlp_sizes) - 2:
                top_layers.append(nn.ReLU())
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, x_cat, price):
        emb_list = [self.embedding_layers[field](x_cat[i]) for i, field in enumerate(self.embedding_layers)]
        dense_embed = self.price_proj(price.unsqueeze(1))
        all_features = emb_list + [dense_embed]

        # Pairwise interactions (dot product)
        interactions = []
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                interactions.append(torch.sum(all_features[i] * all_features[j], dim=1, keepdim=True))
        interaction_term = torch.cat(interactions, dim=1)

        dense_input = self.bottom_mlp(dense_embed)
        concat = torch.cat([interaction_term, dense_input], dim=1)
        out = self.top_mlp(concat).squeeze(1)
        return out