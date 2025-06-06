import json
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np


encoder = LabelEncoder()
encoder.classes_ = np.load('H:\dev\magisterka\gnn-hate-speech-detection\models\hateXplain\classes.npy', allow_pickle=True)

with open('H:\dev\magisterka\gnn-hate-speech-detection\models\hateXplain/dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def get_majority_label(annotators):
    labels = [a['label'] for a in annotators]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0] if most_common else None

texts, labels = [], []

for post_id, post_data in raw_data.items():
    majority_label = get_majority_label(post_data['annotators'])
    if majority_label in encoder.classes_:
        label_id = encoder.transform([majority_label])[0]
        text = ' '.join(post_data['post_tokens']) 
        texts.append(text)
        labels.append(label_id)

test_text = texts[:5]

for i in range(len(test_text)):
    word_count = len(test_text[i].split())
    print(f"Text {i+1} has {word_count} words.")
