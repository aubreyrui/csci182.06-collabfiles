import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class TranslationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TranslationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)  

    def forward(self, input_text):
        embeds = self.embedding(input_text) 
        pooled = embeds.mean(dim=1)  
        output = self.fc(pooled)  
        return output  
    
class TranslationDataset(Dataset):
    def __init__(self, df):
        self.eng_texts = df['English'].apply(text_pipeline).tolist()  

        self.label_mapping = {label: idx for idx, label in enumerate(df['Tagalog'].unique())}
        self.labels = df['Tagalog'].map(self.label_mapping).tolist() 

    def __len__(self):
        return len(self.eng_texts)

    def __getitem__(self, idx):
        return self.eng_texts[idx], self.labels[idx]  
        
csv_path = os.path.expanduser("english_to_tagalog_1000.csv")
df = pd.read_csv(csv_path)

direct_mapping_dict = {eng: tl for eng, tl in zip(df['English'], df['Tagalog'])}
word_to_word_dict = {}

for eng, tl in zip(df['English'], df['Tagalog']):
    eng_words = eng.split()
    tl_words = tl.split()

    if len(eng_words) == len(tl_words):
        for e_word, t_word in zip(eng_words, tl_words):
            word_to_word_dict[e_word] = t_word

def build_vocab(texts):
    word_counter = Counter()
    for text in texts:
        words = text.split() 
        word_counter.update(words)

    vocab = {word: idx+1 for idx, (word, index) in enumerate(word_counter.most_common())}
    vocab["<unk>"] = 0  
    return vocab

vocab = build_vocab(df['English'])
vocab_tl = build_vocab(df['Tagalog'])

def text_pipeline(text):
    return torch.tensor([vocab.get(word, vocab["<unk>"]) for word in text.split()], dtype=torch.long)

def chunk_by_punctuation(text):
    chunks = re.split(r'([.!?,])', text)  
    processed_chunks = []

    i = 0
    while i < len(chunks):
        chunk = chunks[i].strip()
        if i + 1 < len(chunks) and chunks[i + 1] in ".!?,":
            chunk += chunks[i + 1]  
            i += 1  
        
        if chunk:  
            processed_chunks.append(chunk)
        i += 1  

    return processed_chunks

def preprocess_chunk(chunk):
    words_only = re.sub(r'[.!?,]', '', chunk).strip()  
    punctuation = re.findall(r'[.!?,]', chunk)  
    return words_only, "".join(punctuation)  

def translate_sentence(model, text, vocab, label_mapping):
    print(f"\nTranslating: {text}\n")

    model.eval()  
    chunks = chunk_by_punctuation(text)  
    print(f"Chunks for translation: {chunks}\n")

    translated_chunks = []
    
    for chunk in chunks:
        words_only, punctuation = preprocess_chunk(chunk)  

        if not words_only:  
            translated_chunks.append(punctuation)
            continue  

        words = words_only.split()
        text_tensor = torch.tensor([vocab.get(word, vocab["<unk>"]) for word in words], dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            output = model(text_tensor)

        predicted_idx = torch.argmax(output, dim=1).item()
        tagalog_translation = {idx: label for label, idx in label_mapping.items()}
        model_translation = tagalog_translation.get(predicted_idx, "Unknown")

        print(f"Model prediction for chunk '{chunk}': {model_translation}\n")
        translated_chunks.append(model_translation + punctuation)

    final_translation = " ".join(translated_chunks)
    print(f"Final Translated Sentence: {final_translation}\n")

    return final_translation

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = TranslationDataset(train_df)
test_dataset = TranslationDataset(test_df)

embed_dim = 128
batch_size = 32
epochs = 30
learning_rate = 0.0055

def collate_batch(batch):
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)

    padded_texts = [torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.long)]) for t in texts]
    return torch.stack(padded_texts), torch.tensor(labels, dtype=torch.long)  

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

vocab_size = len(vocab)
num_classes = len(df['Tagalog'].unique()) 
model = TranslationModel(vocab_size, embed_dim, num_classes)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model.forward(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("Training complete!")

def evaluate(model, dataloader):
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in dataloader:
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    model.train() 

evaluate(model, test_loader)

sample_text1 = "Hello"
sample_text2 = "Hello, how are you?"
sample_text3 = "Good morning"
sample_text4 = "I love you"
sample_text5 = '''
Good morning! I know it's a Sunday, but I just want to tell you I love you. 
How are you? I don't understand why you won't text me back. I'm sorry, okay?
Can we be friends again? You wont have to call the police next time. I promise.'''

translation1 = translate_sentence(model, sample_text1, vocab, train_dataset.label_mapping)
translation2 = translate_sentence(model, sample_text2, vocab, train_dataset.label_mapping)
translation3 = translate_sentence(model, sample_text3, vocab, train_dataset.label_mapping)
translation4 = translate_sentence(model, sample_text4, vocab, train_dataset.label_mapping)
translation5 = translate_sentence(model, sample_text5, vocab, train_dataset.label_mapping)

print(f"English: {sample_text1}")
print(f"Tagalog Translation: {translation1}")

print(f"English: {sample_text2}")
print(f"Tagalog Translation: {translation2}")

print(f"English: {sample_text3}")
print(f"Tagalog Translation: {translation3}")

print(f"English: {sample_text4}")
print(f"Tagalog Translation: {translation4}")

print(f"English: {sample_text5}")
print(f"Tagalog Translation: {translation5}")

'''
if you want to see direct mapping
output_file = "new_mappings_output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("--- Direct Phrase Mappings ---\n")
    for eng, tl in direct_mapping_dict.items():
        f.write(f"{eng} -> {tl}\n")

    f.write("\n--- Word-to-Word Mappings ---\n")
    for eng_word, tl_word in word_to_word_dict.items():
        f.write(f"{eng_word} -> {tl_word}\n")

print(f"Mappings saved to {output_file}")
'''

