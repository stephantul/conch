"""Create concept vectors."""
import numpy as np
import json

from tqdm import tqdm
from reach import Reach
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer


def create_concepts(concepts,
                    embeddings,
                    include_np=True,
                    apply_idf=False,
                    max_df=1.0):
    """Create concepts by summing over descriptions in embedding spaces."""
    # Gold standard labels for concepts:
    sty = json.load(open("data/concept_label.json"))

    concept_names = []
    vectors = []

    concept_labels = []

    descs = [" ".join(v).lower() for v in concepts.values()]
    if apply_idf:
        t = TfidfVectorizer(max_df=max_df)
        t.fit(descs)
        embeddings = embeddings.intersect(set(t.vocabulary_))

    for name, descriptions in tqdm(concepts.items()):

        try:
            label = sty[name]
        except KeyError:
            continue

        if not include_np and label == "np":
            continue

        descs = " ".join(descriptions).lower().split()
        try:
            embs = embeddings.vectorize(descs, remove_oov=True)
        except ValueError:
            continue
        v = np.mean(embs, 0)

        if np.all(v == 0):
            continue

        vectors.append(v)
        concept_labels.append(label)
        name = "{0}_{1}".format(name, "_".join(descriptions[0].split()))
        concept_names.append(name)

    r = Reach(np.array(vectors), concept_names)
    name2label = dict(zip(concept_names, concept_labels))

    return r, name2label


if __name__ == "__main__":

    path_to_embeddings = ""
    r_1 = Reach.load("../../corpora/mimiciii-min5-neg3-w5-100.vec",
                     unk_word="UNK")

    concepts = json.load(open("data/all_concepts.json"))
    r, name2label = create_concepts(concepts, r_1, include_np=True)

    r.save_fast_format("data/concept_vectors")
    json.dump(name2label, open("data/names2label.json", 'w'))
