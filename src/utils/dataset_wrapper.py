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
        text = self.texts[idx]
        label = self.labels[idx]
        graph_data = self.text_to_graph(text)
        graph_data.y = torch.tensor(label, dtype=torch.long)
        return graph_data
