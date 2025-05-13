import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, texts, labels, text_to_graph):
        self.texts = texts
        self.labels = labels
        self.text_to_graph = text_to_graph

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        data = self.text_to_graph(self.texts[idx])
        data.y = torch.tensor(self.labels[idx], dtype=torch.long)
        return data
