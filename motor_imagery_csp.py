
import numpy as np  
import matplotlib.pyplot as plt  
import mne  
from mne.channels import make_standard_montage  
from mne.datasets import eegbci 
from mne.decoding import CSP  
from mne.io import concatenate_raws, read_raw_edf  
from mne.viz import plot_topomap  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn.model_selection import ShuffleSplit, cross_val_score 
from sklearn.pipeline import Pipeline  


def main() -> None:
    """Tai du lieu, tien xu ly, epoch, huan luyen va danh gia mo hinh."""

    # --- 1. Tai du lieu ----------------------------------------------------
    subject = 1  
    runs = [4, 8, 12]  
    raw_paths = eegbci.load_data(subject, runs)

    # Doc tung file EDF va noi thanh mot raw duy nhat (preload de ghi vao RAM)
    raws = [read_raw_edf(path, preload=True, stim_channel="auto") for path in raw_paths]
    raw = concatenate_raws(raws)  # Noi cac doan run lai thanh mot Raw duy nhat

    # --- 2. Tien xu ly -----------------------------------------------------
    montage = make_standard_montage("standard_1020")
    standard_lookup = {name.upper(): name for name in montage.ch_names}
    raw.rename_channels(lambda x: standard_lookup.get(x.rstrip(".").upper(), x.rstrip(".")))
    raw.set_montage(montage)

    # Loc thong day 7-30 Hz: Mu + Beta
    raw.filter(l_freq=7.0, h_freq=30.0, fir_design="firwin", verbose=False)

    # Chi lay kenh EEG, bo EOG/stim
    raw.pick_types(eeg=True, eog=False, stim=False)

    # --- 3. Epoching -------------------------------------------------------
    events, event_id_map = mne.events_from_annotations(raw)

    rest_id = event_id_map.get("T0", 1)
    left_id = event_id_map.get("T1", 2)
    right_id = event_id_map.get("T2", 3)

    event_id = {"left_hand": left_id, "right_hand": right_id}
    tmin, tmax = 0.0, 4.0

    # Epoch MI (dung cho train CSP + LDA)
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

    X = epochs.get_data()
    y = epochs.events[:, -1]  # 2 = left, 3 = right

    # --- 3b. Topomap cong suat & ERD tu data goc (truoc CSP) ---------------
    # Epoch REST (T0) cung cua so 0–4 s
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

    # PSD 8–30 Hz
    psd_rest = epochs_rest["rest"].compute_psd(fmin=8, fmax=30, method="welch")
    psd_left = epochs["left_hand"].compute_psd(fmin=8, fmax=30, method="welch")
    psd_right = epochs["right_hand"].compute_psd(fmin=8, fmax=30, method="welch")

    # Trung binh theo epoch & tan so -> 1 gia tri / kenh
    rest_power = psd_rest.get_data().mean(axis=(0, 2))
    left_power = psd_left.get_data().mean(axis=(0, 2))
    right_power = psd_right.get_data().mean(axis=(0, 2))

    # ERD: Rest - MI (duong = giam cong suat khi MI)
    erd_left = rest_power - left_power
    erd_right = rest_power - right_power
    diff_lr = left_power - right_power  # Left - Right (tong cong suat)

    # ---- 3c. KIỂM TRA BẰNG BIỂU ĐỒ Ở C3/C4 -------------------------------
    motor_chs = ["C3", "C4"]
    picks_motor = mne.pick_channels(epochs.info["ch_names"], motor_chs)

    motor_rest = rest_power[picks_motor]
    motor_left = left_power[picks_motor]
    motor_right = right_power[picks_motor]
    motor_erd_left = erd_left[picks_motor]
    motor_erd_right = erd_right[picks_motor]

    # Biểu đồ 1: Rest vs Left vs Right tại C3/C4
    x = np.arange(len(motor_chs))
    width = 0.25

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    ax_bar.bar(x - width, motor_rest, width, label="Rest")
    ax_bar.bar(x, motor_left, width, label="Left MI")
    ax_bar.bar(x + width, motor_right, width, label="Right MI")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(motor_chs)
    ax_bar.set_ylabel("Power 8–30 Hz")
    ax_bar.set_title("Rest vs Left vs Right\n(power tại C3/C4)")
    ax_bar.legend(loc="best")

    # Biểu đồ 2: ERD (Rest-Left, Rest-Right) tại C3/C4
    fig_erd, ax_erd = plt.subplots(figsize=(6, 4))
    ax_erd.bar(x - width / 2, motor_erd_left, width, label="ERD Left (Rest-Left)")
    ax_erd.bar(x + width / 2, motor_erd_right, width, label="ERD Right (Rest-Right)")
    ax_erd.set_xticks(x)
    ax_erd.set_xticklabels(motor_chs)
    ax_erd.set_ylabel("Rest - MI (8–30 Hz)")
    ax_erd.set_title("ERD tại C3/C4")
    ax_erd.axhline(0, color="k", linewidth=0.8)
    ax_erd.legend(loc="best")

    print("=== Power 8–30 Hz tại C3/C4 ===")
    for ch, r, l, ri in zip(motor_chs, motor_rest, motor_left, motor_right):
        print(f"{ch}: Rest={r:.3f}, Left={l:.3f}, Right={ri:.3f}")
    print("=== ERD (Rest-MI) tại C3/C4 ===")
    for ch, el, er in zip(motor_chs, motor_erd_left, motor_erd_right):
        print(f"{ch}: ERD Left={el:.3f}, ERD Right={er:.3f}")

    # ---- 3d. Topomap ERD & Left-Right ------------------------------------
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

    components_to_plot = list(range(min(8, csp.filters_.shape[0])))
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
