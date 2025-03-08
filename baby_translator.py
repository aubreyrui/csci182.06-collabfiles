import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextProcessor:
    def __init__(self, csv_path, num_dim):
        self.csv_path = csv_path
        self.num_dim = num_dim
        self.df = pd.read_csv(csv_path)

        self.eng_to_tag = {row['English'].lower(): row['Tagalog'].lower() for _, row in self.df.iterrows()}

        full_text = " ".join(self.df['Tagalog'].dropna().astype(str))
        self.vocabulary = self.create_vocabulary(full_text)
        self.vocab_size = len(self.vocabulary)

        print("Vocabulary (first 100 words):", self.vocabulary[:100])
        print("Vocabulary size:", self.vocab_size)

    def create_vocabulary(self, text):
        output = text.replace('.', ' <period>')
        output = output.replace('!', ' <exclamation>')
        output = output.replace(',', ' <comma>')
        output = output.lower()
        output = output.split()
        return sorted(list(set(output)))

    def encode(self, text):
        words = text.lower().split()
        encode_out = {word: i for i, word in enumerate(self.vocabulary)}
        return [encode_out.get(word, 0) for word in words]  

    def decode(self, indices):
        decode_out = {i: word for i, word in enumerate(self.vocabulary)}
        return [decode_out.get(index, "[UNK]") for index in indices]

    def get_direct_translation(self, sentence):
        words = sentence.lower().split()
        translated_text = []

        i = 0
        while i < len(words):
            found = False
            for j in range(len(words), i, -1):  
                phrase = " ".join(words[i:j])
                if phrase in self.eng_to_tag:
                    translated_text.append(self.eng_to_tag[phrase])
                    i = j
                    found = True
                    break
            if not found:
                translated_text.append(words[i])
                i += 1

        return " ".join(translated_text) if translated_text else None

class TranslationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TranslationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, input_text):
        embedded = self.embedding(input_text)
        hidden = self.relu(self.linear1(embedded))
        hidden = self.relu(self.linear2(hidden))
        output = self.linear3(hidden)
        return output

def translator(model, processor, sentence):
    words = sentence.lower().split()
    translated_words = []
    i = 0
    while i < len(words):
        found = False
        for j in range(len(words), i, -1):
            phrase = " ".join(words[i:j])
            if phrase in processor.eng_to_tag:
                translated_words.append(processor.eng_to_tag[phrase])
                i = j
                found = True
                break
        if not found:
            translated_words.append(words[i]) 
            i += 1

    untranslated_indices = [i for i, word in enumerate(translated_words) if word not in processor.eng_to_tag.values()]
    
    if untranslated_indices:
        untranslated_words = [translated_words[i] for i in untranslated_indices]
        tokenized = processor.encode(" ".join(untranslated_words))
        tensor_input = torch.tensor([tokenized])
        model_output = model(tensor_input).argmax(dim=-1).squeeze()
   
        if model_output.dim() == 0:
            model_output = [model_output.item()]
        else:
            model_output = model_output.tolist()
        model_translated = processor.decode(model_output)

        for i, translated_word in zip(untranslated_indices, model_translated):
            translated_words[i] = translated_word

    return " ".join(translated_words)

if __name__ == "__main__":
    csv_path = os.path.expanduser("~/Desktop/NLP Midterms/english_to_tagalog_1000.csv")
    processor = TextProcessor(csv_path, num_dim=5)

    model = TranslationModel(len(processor.vocabulary), embedding_dim=5, hidden_dim=3)

    print("Translated:", translator(model, processor, "Hello, how are you?"))
    print("Translated:", translator(model, processor, "Good morning"))
    print("Translated:", translator(model, processor, "I love you"))
    print("Translated:", translator(model, processor, "Come here"))
    print("Translated:", translator(model, processor, "What is your name?"))
    print("Translated:", translator(model, processor, "Where are you going?"))

    print("Translated:", translator(model, processor, 
                                    '''Monday left me broken
                                        Tuesday, I was through with hoping
                                        Wednesday, my empty arms were open
                                        Thursday, waiting for love, waiting for love
                                    '''))
