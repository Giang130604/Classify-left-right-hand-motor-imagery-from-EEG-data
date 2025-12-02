"""Fit CSP on full data and plot spatial patterns (topomap)."""

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    clf.fit(X, y)
    csp = clf.named_steps["csp"]

    components_to_plot = list(range(min(4, csp.filters_.shape[0])))
    csp.plot_patterns(
        epochs.info,
        ch_type="eeg",
        units="Patterns (a.u.)",
        size=2.0,
        components=components_to_plot,
        contours=6,
        outlines="head",
    )
    plt.show()


if __name__ == "__main__":
    main()
