"""
Extrinsic evaluation on the i2b2 data.

Experiment 3 in the paper.
Set the boolean flag Perfect to True if you want to try perfect chunking.
"""
import numpy as np
import json

from reach import Reach
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from conch.evaluation.sequence import precision_recall_dict, eval_sequence
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans


def rewrite_labels(seq):
    out = ["O" if seq[0] == "np" else f"B-{seq[0].upper()}"]
    prev = out[-1]
    for x in seq[1:]:
        if x == "np":
            out.append("O")
        elif prev != "O":
            label = prev.split("-")[1]
            if x == label:
                out.append(f"I-{x}")
            else:
                out.append(f"B-{x}")
        else:
            out.append(f"B-{x}")
        prev = out[-1]
    return out


def distance(x, axis=1):
    assert np.allclose(x.sum(axis), 1)
    h_uniform = entropy(np.full(x.shape[axis], 1 / x.shape[axis]), 0)
    h_x = entropy(x, axis)
    return h_uniform - h_x


def entropy(x, axis=1):
    """Entropy of a 2D array."""
    return np.sum(-x * np.log2(x), axis)


def softmax(x, axis=1):
    x = x - x.max(axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis, keepdims=True)


def similarity(x, y, metric="dot", pooling=np.mean, **kwargs):
    if metric == "dot":
        return pooling(x.dot(v.T), axis=1)
    if metric == "rbf":
        return pooling(rbf_kernel(x, y, **kwargs), axis=1)
    else:
        return pooling(1 - pairwise_distances(x, y, metric=metric), axis=1)


if __name__ == "__main__":

    # Set this flag to true to replicate the perfect chunking setting
    # in experiment 3.
    perfect = False

    gold = json.load(open("data/test_gold.json"))
    gold = list(zip(*sorted(gold.items())))[1]

    if perfect:
        data = json.load(open("data/test_gold.json"))
    else:
        data = json.load(open("data/test_uima.json"))
    data = list(zip(*sorted(data.items())))[1]

    txt, gold_bio = zip(*gold)
    _, data_bio = zip(*data)

    r = Reach.load("../../corpora/mimiciii-min5-neg3-w5-100.vec",
                   unk_word="<UNK>")

    r_concept = Reach.load_fast_format(f"data/concept_vectors")
    concept_labels = json.load(open("data/names2label.json"))

    grouped = defaultdict(list)
    for k, v in concept_labels.items():
        grouped[v].append(r_concept[k])

    grouped.pop("np")

    memory = {}
    for k, v in tqdm(grouped.items()):

        km = KMeans(10)
        km.fit(v)
        memory[k] = km.cluster_centers_

    scores = []
    vecs = r.transform([" ".join(x).lower().split() for x in txt])

    preds = []
    for x in tqdm(vecs):
        atts = []
        ks, vs = zip(*memory.items())

        for v in vs:
            atts.append(similarity(x, v, "rbf", pooling=np.max))
        atts = np.stack(atts, 1)
        score = softmax(atts, 1)
        pred = np.asarray([ks[x] for x in np.argmax(score, 1)])

        dist = distance(score)
        mask = dist < dist.mean()
        pred[mask] = "np"
        preds.append(rewrite_labels(pred))

    s = eval_sequence(preds, gold_bio, exact=False)
    a = precision_recall_dict(*s, average="micro")
    b = precision_recall_dict(*s, average="macro")
    scores.append((a, b))
