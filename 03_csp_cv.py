"""Run CSP + LDA with cross-validation and report accuracy."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne.decoding import CSP

from bci_utils import load_and_preprocess_raw, make_epochs


def main() -> None:
    raw = load_and_preprocess_raw(subject=1, runs=(4, 8, 12))
    epochs, _ = make_epochs(raw, tmin=0.0, tmax=4.0)

    X = epochs.get_data()
    y = epochs.events[:, -1]

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("csp", csp), ("lda", lda)])

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    print(f"Mean accuracy (10x ShuffleSplit): {np.mean(scores):.3f}")
    print("All scores:", np.round(scores, 3))


if __name__ == "__main__":
    main()
