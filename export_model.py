#!/usr/bin/env python3
"""Export the trained TF-IDF vectorizer + logistic regression to docs/model.json for the website.

Run from the repo root after the notebook has written models/*.joblib:

    python export_model.py                      # defaults
    python export_model.py --max-terms 20000    # smaller file, slightly less accurate

Needs: joblib, numpy, nltk (same install as the notebook). scikit-learn must be importable to unpickle the models.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def nltk_helpers():
    """Stop words and the exact lemmatizer the notebook used."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)

    lem = WordNetLemmatizer()

    def lemma(w):
        return lem.lemmatize(lem.lemmatize(w, "v"), "n")      # same two-step call as the notebook

    # Irregular forms WordNet lists (ran, children, ...). Uses a private NLTK attribute, so it is guarded.
    irregular = set()
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        irregular = {k for pos in ("v", "n") for k in wn._exception_map.get(pos, {})}
    except Exception as e:
        print("note: could not read WordNet irregular forms:", e)
    return sorted(set(stopwords.words("english"))), lemma, irregular


def inflections(w):
    """Common surface forms of a base word: hearts, loved, running, cities, wolves ..."""
    c = {w + "s", w + "es", w + "ed", w + "d", w + "ing"}
    if w.endswith("e"):
        c.add(w[:-1] + "ing")
    if w.endswith("y"):
        c |= {w[:-1] + "ies", w[:-1] + "ied"}
    if w.endswith("f"):
        c.add(w[:-1] + "ves")
    if w.endswith("fe"):
        c.add(w[:-2] + "ves")
    if len(w) >= 2:
        c |= {w + w[-1] + "ing", w + w[-1] + "ed"}
    return c


def build_lemma_map(terms, lemma, irregular, stop):
    """surface form -> vocabulary term, computed with the real lemmatizer so the browser matches training."""
    termset = set(terms)
    forms = set(irregular)
    for t in terms:
        forms |= inflections(t)
    out = {}
    for n, f in enumerate(sorted(forms), 1):
        if len(f) >= 2 and f not in stop:
            l = lemma(f)
            if l != f and l in termset:
                out[f] = l
        if n % 100_000 == 0:
            print(f"  lemma map: {n:,} / {len(forms):,} forms")
    return out


def main():
    default_models = "model" if (Path("model") / "tfidf_vectorizer.joblib").exists() else "models"
    default_data = "Data" if (Path("Data") / "test_labels.csv").exists() else "data"

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=default_models)
    ap.add_argument("--data", default=default_data)
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--max-terms", type=int, default=40_000,
                    help="keep coefficients for the N terms with the largest absolute weight")
    args = ap.parse_args()

    tfidf = joblib.load(Path(args.models) / "tfidf_vectorizer.joblib")
    clf = joblib.load(Path(args.models) / "logistic_regression.joblib")
    assert type(clf).__name__ == "LogisticRegression", "the site assumes softmax (multinomial) logistic regression"
    assert tfidf.ngram_range == (1, 1) and tfidf.norm == "l2", "the site assumes unigram, L2-normalised TF-IDF"

    terms = tfidf.get_feature_names_out().tolist()
    idf = np.asarray(tfidf.idf_, dtype=np.float64)
    coef = np.asarray(clf.coef_, dtype=np.float64)                  # (classes, terms)
    classes = [str(c) for c in clf.classes_]
    assert coef.shape == (len(classes), len(terms))

    # Keep coefficients only for the strongest terms. idf is kept for ALL terms because the
    # L2 normalisation of a song's vector uses every in-vocabulary word.
    strength = np.abs(coef).max(axis=0)
    keep = np.sort(np.argsort(strength)[::-1][: args.max_terms])
    coef_kept = np.round(coef[:, keep].T, 4)                        # (kept terms, classes)
    intercept = np.round(clf.intercept_, 4)

    print(f"{len(terms):,} terms in vocabulary, keeping coefficients for {len(keep):,}")
    stop, lemma, irregular = nltk_helpers()
    print("building lemma map (a minute or two)...")
    lemma_map = build_lemma_map(terms, lemma, irregular, set(stop))

    payload = {
        "classes": classes,
        "intercept": intercept.tolist(),
        "terms": terms,
        "idf": np.round(idf, 5).tolist(),
        "coef_index": keep.tolist(),
        "coef": coef_kept.ravel().tolist(),
        "stop": stop,
        "lemma": lemma_map,
        "sublinear_tf": bool(tfidf.sublinear_tf),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(payload, separators=(",", ":"))
    out.write_text(json_str)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {len(lemma_map):,} lemma entries")
    if out.resolve() != (Path("docs") / "model.json").resolve():
        docs_out = Path("docs") / "model.json"
        docs_out.parent.mkdir(parents=True, exist_ok=True)
        docs_out.write_text(json_str)
        print(f"also synced to {docs_out}")

    # How much does pruning + rounding cost on the held-out test set?
    xt, yt = Path(args.data) / "X_test_tfidf.npz", Path(args.data) / "test_labels.csv"
    if xt.exists() and yt.exists():
        import pandas as pd
        import scipy.sparse as sp
        X_test = sp.load_npz(xt)
        y_test = pd.read_csv(yt, keep_default_na=False)["genre"].to_numpy()
        W = np.zeros_like(coef)
        W[:, keep] = np.round(coef[:, keep], 4)
        logits = np.asarray(X_test @ W.T) + intercept
        pruned = np.array(classes)[logits.argmax(axis=1)]
        print(f"test accuracy: full model {(clf.predict(X_test) == y_test).mean():.4f} | "
              f"exported (pruned + rounded) {(pruned == y_test).mean():.4f}")


if __name__ == "__main__":
    main()
