import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

from GNNs.GCNClassifier import GCNClassifier
from GNNs.GRNClassifier import GRNClassifier

from embeddings.word2vec import Word2VecEmbedding
from embeddings.sbert import SBERTEmbedding

from utils.graph_of_words import GraphOfWords
from utils.graph_to_data import GraphToData
from utils.dataset_wrapper import DatasetWrapper


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Replace with HateXplain dataset
    texts = [
        "This is a hateful comment.",
        "You are amazing!",
        "Such a toxic attitude.",
        "I love this movie.",
        "You suck and should go away.",
        "What a wonderful day!",
    ]
    labels = [1, 0, 1, 0, 1, 0]

    w2v_embedder = Word2VecEmbedding("./models/google/GoogleNews-vectors-negative300.kv", device=device)
    sbert_embedder = SBERTEmbedding(device=device)
    gow = GraphOfWords(embedding_model=sbert_embedder, window_size=2)
    text_to_graph = GraphToData(gow)

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.33, random_state=42)
    train_dataset = DatasetWrapper(X_train, y_train, text_to_graph)
    test_dataset = DatasetWrapper(X_test, y_test, text_to_graph)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    model = GCNClassifier(in_channels=384, hidden_channels=128, num_classes=2).to(device) # Adjust input size when changing embedder!!!!
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()

    for epoch in range(1, 11):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2%}")

if __name__ == "__main__":
    main()
