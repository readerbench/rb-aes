import argparse
import csv
import math
import os
import random
from heapq import heapify, heappop
from typing import Counter, Dict, List, Tuple

from joblib import Parallel, delayed
from rb import Document, Lang
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from scipy.stats import f_oneway, pearsonr


def load_dataset(filename: str) -> Tuple[List[str], List[float]]:
    texts, features, labels = [], [], []
    with open(filename, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            text = row[0]
            label = float(row[-1])
            f = [float(x) for x in row[1:-1]]
            features.append(f)
            texts.append(text)
            labels.append(float(label))
    return texts, features, labels

def split_dataset(texts: List[str], labels: List[float], val_ratio=0.1, test_ratio=0.1):
    indices = list(range(len(texts)))
    random.shuffle(indices)
    num_train = int((1-val_ratio-test_ratio) * len(texts))
    num_val = int(val_ratio * len(texts))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:(num_train+num_val)]
    test_indices = indices[(num_train+num_val):]
    train_texts = [texts[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    val_texts = [texts[index] for index in val_indices]
    val_labels = [labels[index] for index in val_indices]
    test_texts = [texts[index] for index in test_indices]
    test_labels = [labels[index] for index in test_indices]
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    
def compute_features(doc: Document):
    model = create_vector_model(doc.lang, VectorModelType.TRANSFORMER, None)
    cna_graph = CnaGraph(docs=doc, models=[model])
    compute_indices(doc=doc, cna_graph=cna_graph)     
    return {
        str(feature): value
        for feature, value in doc.indices.items()
    }
        
def construct_documents(texts: List[str], lang: Lang) -> List[Dict[ComplexityIndex, float]]:
    docs = [Document(lang, text) for text in texts]
    create_vector_model(lang, VectorModelType.TRANSFORMER, None)
    return Parallel(n_jobs=-1, prefer="processes")( \
        delayed(compute_features)(doc) \
        for doc in docs)


def filter_rare(features: List[str], docs: List[Dict[str, float]]) -> List[str]:
    result = []
    for feature in features:
        values = [doc[feature] for doc in docs]
        counter = Counter(values)
        if counter.most_common(1)[0][1] < 0.2 * len(values):
            result.append(feature)
    return result

def correlation_with_targets(feature: str, docs: List[Dict[str, float]], labels: List[float]) -> float:
    values = [doc[feature] for doc in docs]
    corr, p = pearsonr(values, labels)
    return abs(corr)
   

def remove_colinear(features: List[str], docs: List[Dict[str, float]], labels: List[float]) -> List[str]:
    heap = []
    for i, a in enumerate(features[:-1]):
        for j in range(i+1, len(features)):
            b = features[j]
            values_a = [doc[a] for doc in docs]
            values_b = [doc[b] for doc in docs]
            corr, p = pearsonr(values_a, values_b)
            if math.isnan(corr):
                continue
            heap.append((-corr, i, j))
    heapify(heap)
    
    correlations = [
        correlation_with_targets(feature, docs, labels) 
        for feature in features
    ]
    mask = [True] * len(features)
    while len(heap) > 0:
        inv_corr, i, j = heappop(heap)
        if not mask[i] or not mask[j]:
            continue
        if inv_corr < -0.9:
            if correlations[i] > correlations[j]:
                mask[j] = False
            else:
                mask[i] = False
    return [
        feature
        for feature, m in zip(features, mask)
        if mask
    ]

def save_dataset(features: List[str], texts: List[str], docs: List[Dict[str, float]], labels: List[float], filename: str):
    with open(filename, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["Text"] + features + ["Label"])
        for text, doc, label in zip(texts, docs, labels):
            writer.writerow([text] + [doc[feature] for feature in features] + [label])

def prepare(filename: str, lang: Lang, root: str):
    os.makedirs(root, exist_ok=True)
    texts, _, labels = load_dataset(filename)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_dataset(texts, labels)
    train_indices = construct_documents(train_texts, lang)
    features = list(sorted({feature for doc in train_indices for feature in doc}))
    features = filter_rare(features, train_indices)
    features = remove_colinear(features, train_indices, train_labels)
    save_dataset(features, train_texts, train_indices, train_labels, f"{root}/train.csv")
    val_indices = construct_documents(val_texts, lang)
    save_dataset(features, val_texts, val_indices, val_labels, f"{root}/val.csv")
    test_indices = construct_documents(test_texts, lang)
    save_dataset(features, test_texts, test_indices, test_labels, f"{root}/test.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess and split dataset')
    parser.add_argument('--lang', default="ro", choices=["ro", "en", "fr"], type=str,
                    help='language')
    parser.add_argument("--file", required=True, type=str, help="Name of csv file")
    parser.add_argument("--dest", required=True, type=str, help="Name of output folder")
    args = parser.parse_args()
    lang = Lang[args.lang.upper()]
    prepare(args.file, lang, args.dest)
    
    
    