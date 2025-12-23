import torch

import numpy as np
import pandas as pd

hero_features = pd.read_excel("./data/hero_features.xlsx")
hero_semantic_embeddings = torch.load("./data/hero_semantic_embeddings.pt")
HERO_ID_FEATURE_MAP = {
    row['id']: torch.Tensor(row.drop(labels=['index', 'name', 'id']).values.astype(np.float32))
    for _, row in hero_features.iterrows()
}
HERO_ID_SEMANTIC_MAP = {
    row['id']: hero_semantic_embeddings[row['name']]
    for _, row in hero_features.iterrows()
}