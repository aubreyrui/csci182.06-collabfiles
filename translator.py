import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from collections import Counter

class EngTagTranslationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EngTagTranslationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_layer_1 = nn.Linear(embedding_dim, 5)
        self.output_layer = nn.Linear(5, 5)

    def build_vocab(texts):
        word_counter = Counter()
        for text in texts:
            text = text.replace('!', ' <exclaim>')
            text = text.replace('.', ' <period>')
            text = text.replace('?', ' <question>')
            text = text.lower()
            words = text.split()
            word_counter.update(words)

        vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counter.most_common())}
        vocab["<unk>"] = 0  # Assign index 0 to unknown words
        return vocab
 
df = pd.read_csv("english_to_tagalog_1000.csv")
en_vocab = EngTagTranslationModel.build_vocab(df["English"].tolist())
print(en_vocab)
tag_vocab = EngTagTranslationModel.build_vocab(df["Tagalog"].tolist())
print(tag_vocab)