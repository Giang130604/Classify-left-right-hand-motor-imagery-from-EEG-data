from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from bci_utils import load_and_preprocess_raw, make_epochs

# Store/download EEGBCI data inside the project (avoid C: drive)
PROJECT_DIR = Path(__file__).resolve().parent
MNE_DATA_DIR = PROJECT_DIR / "mne_data"
mne.set_config("MNE_DATA", str(MNE_DATA_DIR), set_env=True)
mne.set_config("MNE_DATASETS_EEGBCI_PATH", str(MNE_DATA_DIR), set_env=True)
RESAMPLE_SFREQ = 160.0


def load_subject_epochs(
    subject: int,
    runs: Sequence[int],
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray, mne.Epochs]:
    """Load one subject and return epochs + labels (0=left, 1=right)."""
    raw = load_and_preprocess_raw(subject=subject, runs=runs, resample_sfreq=RESAMPLE_SFREQ)
    epochs, event_id = make_epochs(raw, tmin=tmin, tmax=tmax)

    missing = [key for key in ("left_hand", "right_hand") if key not in event_id]
    if missing:
        raise ValueError(f"Subject {subject} missing events: {missing}")

    label_map = {event_id["left_hand"]: 0, event_id["right_hand"]: 1}
    y = np.array([label_map[ev] for ev in epochs.events[:, -1]], dtype=int)
    X = epochs.get_data()
    return X, y, epochs


def build_dataset(
    subjects: Iterable[int],
    runs: Sequence[int],
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], mne.Epochs]:
    """
    Load multiple subjects and stack trials.

    Returns
    -------
    X, y, groups : stacked data/labels/subject-ids per trial
    loaded_subjects : subjects that were used
    skipped_subjects : subjects skipped due to errors
    example_epochs : epochs from the first loaded subject (for plotting)
    """
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    loaded: List[int] = []
    skipped: List[int] = []
    example_epochs: mne.Epochs | None = None

    for subject in subjects:
        try:
            X_sub, y_sub, epochs_sub = load_subject_epochs(subject, runs, tmin, tmax)
        except Exception as exc:  # noqa: BLE001 - keep reason visible
            print(f"Skip subject {subject}: {exc}")
            skipped.append(subject)
            continue

        X_all.append(X_sub)
        y_all.append(y_sub)
        groups.append(np.full(len(y_sub), subject, dtype=int))
        loaded.append(subject)
        if example_epochs is None:
            example_epochs = epochs_sub

    if not X_all:
        raise RuntimeError("No subject data loaded.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    group_labels = np.concatenate(groups, axis=0)
    return X, y, group_labels, loaded, skipped, example_epochs


def split_subjects(
    subjects: Sequence[int],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Shuffle subjects and split into train/test lists."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(subjects)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_subjects = sorted(shuffled[:n_test].tolist())
    train_subjects = sorted(shuffled[n_test:].tolist())
    if not train_subjects:
        raise ValueError("Train split is empty; reduce test_ratio or add subjects.")
    return train_subjects, test_subjects


def main() -> None:
    runs = [4, 8, 12]
    tmin, tmax = 0.5, 3.5
    n_subjects = 100
    test_ratio = 0.2  # 80/20 subject-level split

    MNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"EEGBCI data directory: {MNE_DATA_DIR}")

    subjects = list(range(1, n_subjects + 1))
    train_subjects, test_subjects = split_subjects(subjects, test_ratio=test_ratio, seed=42)
    print(f"Train subjects ({len(train_subjects)}): {train_subjects[:5]} ...")
    print(f"Test subjects  ({len(test_subjects)}): {test_subjects[:5]} ...")

    X_train, y_train, groups_train, used_train, skipped_train, example_epochs = build_dataset(
        train_subjects, runs, tmin, tmax
    )
    X_test, y_test, _, used_test, skipped_test, _ = build_dataset(test_subjects, runs, tmin, tmax)

    print(f"Loaded train subjects: {len(used_train)} / {len(train_subjects)}")
    if skipped_train:
        print(f"Skipped (train): {skipped_train}")
    print(f"Loaded test subjects: {len(used_test)} / {len(test_subjects)}")
    if skipped_test:
        print(f"Skipped (test): {skipped_test}")

    clf = Pipeline(
        [
            ("csp", CSP(n_components=6, reg='ledoit_wolf', log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )

    cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, groups=groups_train, n_jobs=1)
    print(f"Mean CV accuracy (grouped by subject): {np.mean(cv_scores):.3f}")

    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Held-out subject test accuracy: {test_score:.3f}")

    if example_epochs is not None:
        csp = clf.named_steps["csp"]
        components_to_plot = list(range(min(4, csp.filters_.shape[0])))
        csp.plot_patterns(
            example_epochs.info,
            ch_type="eeg",
            units="Patterns (a.u.)",
            size=1.5,
            components=components_to_plot,
        )
        plt.show()


if __name__ == "__main__":
    main()
