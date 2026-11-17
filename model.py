"""ID＋名称拼接，症状和药品分别经过两层全连接后做内积。"""
from torch import cat, nn
from torch.nn import functional as F


class HerbRecommender(nn.Module):
    def __init__(self, symptom_names, herb_names, dim=32):
        super().__init__()
        self.symptom_embedding = nn.Embedding(len(symptom_names) + 1, dim, padding_idx=0)
        self.herb_embedding = nn.Embedding(len(herb_names), dim)
        self.register_buffer('symptom_names', F.pad(symptom_names, (0, 0, 1, 0)))
        self.register_buffer('herb_names', herb_names)
        self.symptom_tower = nn.Sequential(
            nn.Linear(dim + symptom_names.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, dim),
        )
        self.herb_tower = nn.Sequential(
            nn.Linear(dim + herb_names.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, dim),
        )

    def forward(self, symptoms):
        mask = symptoms.ne(0).unsqueeze(-1)
        symptom_vectors = cat((self.symptom_embedding(symptoms), self.symptom_names[symptoms]), dim=-1)
        pooled = (symptom_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        prescription_vectors = self.symptom_tower(pooled)
        herb_vectors = self.herb_tower(cat((self.herb_embedding.weight, self.herb_names), dim=-1))
        return prescription_vectors @ herb_vectors.T
