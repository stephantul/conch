"""
Extrinsic evaluation on the i2b2 data.

Experiment 3 in the paper.
Set the boolean flag Perfect to True if you want to try perfect chunking.
"""
import numpy as np
import json

from itertools import chain
from conch.evaluation.extrinsic import eval_extrinsic
from conch.preprocessing.baseline import baseline
from conch.preprocessing.concept_vectors import create_concepts
from reach import Reach
from conch.conch import compose, reciprocal
from conch.evaluation.utils import to_conll


if __name__ == "__main__":

    # Set this flag to true to replicate the perfect chunking setting
    # in experiment 3.
    perfect = True

    gold = json.load(open("data/test_gold.json"))
    gold = list(zip(*sorted(gold.items())))[1]

    if perfect:
        data = json.load(open("data/test_gold.json"))
    else:
        data = json.load(open("data/test_uima.json"))
    data = list(zip(*sorted(data.items())))[1]

    txt, gold_bio = zip(*gold)
    _, data_bio = zip(*data)

    embeddings = Reach.load("../../corpora/mimiciii-min5-neg3-w5-100.vec",
                            unk_word="<UNK>")

    concept_reach = Reach.load_fast_format("data/concept_vectors")
    concept_labels = json.load(open("data/names2label.json"))

    gold_bio = list(chain.from_iterable(gold_bio))

    results_bio = {}

    r_phrases = compose(data,
                        f1=np.mean,
                        f2=np.mean,
                        window=0,
                        embeddings=embeddings,
                        context_function=reciprocal)

    pred_bio_focus = eval_extrinsic(list(chain.from_iterable(data_bio)),
                                    r_phrases,
                                    concept_reach,
                                    concept_labels,
                                    250)
