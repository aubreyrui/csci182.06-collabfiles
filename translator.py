import pandas as pd
import torch
import numpy as nn

class EngFilTranslationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EngFilTranslationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_layer_1 = nn.Linear(embedding_dim, 5)
        self.output_layer = nn.Linear(5, 5)

 
df = pd.read_csv("english_to_tagalog_1000.csv")
print(df)