"""Phan loai tuong tuong van dong (tay trai vs tay phai) tren EEG BCI."""

import numpy as np  # Xu ly mang so, tro cho pipeline sklearn
import matplotlib.pyplot as plt  # Ve topomap CSP
import mne  # Thu vien chinh xu ly EEG
from mne.channels import make_standard_montage  # Lay so do dien cuc 10-20 chuan
from mne.datasets import eegbci  # Bo du lieu PhysioNet BCI
from mne.decoding import CSP  # Thuat toan Common Spatial Patterns
from mne.io import concatenate_raws, read_raw_edf  # Doc va noi cac file EDF
from mne.viz import plot_topomap  # Ve topo tu vector cong suat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Bo phan loai LDA
from sklearn.model_selection import ShuffleSplit, cross_val_score  # Chia tap va tinh CV
from sklearn.pipeline import Pipeline  # Ket hop CSP + LDA thanh pipeline


def main() -> None:
    """Tai du lieu, tien xu ly, epoch, huan luyen va danh gia mo hinh."""

    # --- 1. Tai du lieu ----------------------------------------------------
    subject = 1  # Chon doi tuong so 1 theo yeu cau
    runs = [4, 8, 12]  # Cac run MI (nam tay trai/phai) trong bo EEGBCI
    raw_paths = eegbci.load_data(subject, runs)

    # Doc tung file EDF va noi thanh mot raw duy nhat (preload de ghi vao RAM)
    raws = [read_raw_edf(path, preload=True, stim_channel="auto") for path in raw_paths]
    raw = concatenate_raws(raws)  # Noi cac doan run lai thanh mot Raw duy nhat

    # --- 2. Tien xu ly -----------------------------------------------------
    # Gan montage 10-20 chuan de thong tin vi tri dien cuc dung cho topomap
    montage = make_standard_montage("standard_1020")
    standard_lookup = {name.upper(): name for name in montage.ch_names}
    raw.rename_channels(lambda x: standard_lookup.get(x.rstrip(".").upper(), x.rstrip(".")))
    raw.set_montage(montage)

    # Loc thong day 7-30 Hz: giu lai dai tan so Mu (8-13 Hz) va Beta (13-30 Hz)
    raw.filter(l_freq=7.0, h_freq=30.0, fir_design="firwin", verbose=False)

    # Chi lay kenh EEG, bo EOG/stim de tranh nhiem artifact va giam chieu
    raw.pick_types(eeg=True, eog=False, stim=False)

    # --- 3. Epoching -------------------------------------------------------
    # Su kien duoc ma hoa trong annotations voi cac nhan 'T0' (rest), 'T1' (left), 'T2' (right)
    events, event_id_map = mne.events_from_annotations(raw)

    # ID cac su kien
    rest_id = event_id_map.get("T0", 1)
    left_id = event_id_map.get("T1", 2)
    right_id = event_id_map.get("T2", 3)

    event_id = {"left_hand": left_id, "right_hand": right_id}

    # Cua so epoch tu 0s (bat dau tuong tuong) den 4s (ket thuc cue tuong tuong)
    tmin, tmax = 0.0, 4.0

    # Epoch chi cho left/right (dung cho train CSP + LDA)
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    # Lay du lieu 3D (trial x channel x time) va nhan (su kien cuoi cung)
    X = epochs.get_data()
    y = epochs.events[:, -1]  # 2 = left, 3 = right

    # --- 3b. Topomap cong suat & ERD tu data goc (truoc CSP) ---------------
    # Epoch rieng cho REST (T0) cung cua so 0–4 s
    epochs_rest = mne.Epochs(
        raw,
        events=events,
        event_id={"rest": rest_id},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    # Tinh cong suat 8–30 Hz bang Welch
    psd_rest = epochs_rest["rest"].compute_psd(fmin=8, fmax=30, method="welch")
    psd_left = epochs["left_hand"].compute_psd(fmin=8, fmax=30, method="welch")
    psd_right = epochs["right_hand"].compute_psd(fmin=8, fmax=30, method="welch")

    # Trung binh theo epoch va tan so -> 1 gia tri/kenh
    rest_power = psd_rest.get_data().mean(axis=(0, 2))
    left_power = psd_left.get_data().mean(axis=(0, 2))
    right_power = psd_right.get_data().mean(axis=(0, 2))

    # ERD: Rest - MI (duong = giam cong suat khi tuong tuong -> ERD)
    erd_left = rest_power - left_power
    erd_right = rest_power - right_power
    diff_lr = left_power - right_power  # Left - Right (tong cong suat)

    # Dung chung thang mau cho ERD trai/phai
    vmin_erd = min(erd_left.min(), erd_right.min())
    vmax_erd = max(erd_left.max(), erd_right.max())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    plot_topomap(
        erd_left,
        epochs.info,
        axes=axes[0],
        show=False,
        contours=6,
        cmap="RdBu_r",
        vlim=(vmin_erd, vmax_erd),
    )
    axes[0].set_title("ERD Left: Rest - Left\n(8–30 Hz)")

    plot_topomap(
        erd_right,
        epochs.info,
        axes=axes[1],
        show=False,
        contours=6,
        cmap="RdBu_r",
        vlim=(vmin_erd, vmax_erd),
    )
    axes[1].set_title("ERD Right: Rest - Right\n(8–30 Hz)")

    plot_topomap(
        diff_lr,
        epochs.info,
        axes=axes[2],
        show=False,
        contours=6,
        cmap="RdBu_r",
    )
    axes[2].set_title("Left - Right\n(8–30 Hz power)")
    plt.tight_layout()

    # --- 4. Xay dung pipeline CSP + LDA -----------------------------------
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("csp", csp), ("lda", lda)])

    # --- 5. Danh gia cheo --------------------------------------------------
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    print(f"Do chinh xac trung binh (10-lap ShuffleSplit): {np.mean(scores):.3f}")

    # --- 6. Truc quan hoa CSP ---------------------------------------------
    clf.fit(X, y)
    csp = clf.named_steps["csp"]

    components_to_plot = list(range(min(4, csp.filters_.shape[0])))
    csp.plot_patterns(
        epochs.info,
        ch_type="eeg",
        units="Patterns (a.u.)",
        size=1.5,
        components=components_to_plot,
    )
    plt.show()


if __name__ == "__main__":
    main()
