import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, texts, labels, text_to_graph):
        self.graphs = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            try:
                data = text_to_graph(text)
                if data is not None:
                    data.y = torch.tensor(label, dtype=torch.long)
                    self.graphs.append(data)
                else:
                    print(f"[SKIP] Entry {i}: No valid embeddings")
            except Exception as e:
                print(f"[ERROR] Entry {i}: {e}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

