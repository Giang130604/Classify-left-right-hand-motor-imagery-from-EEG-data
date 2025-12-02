"""Tóm tắt nhanh dataset EEGBCI và hình học kênh/PSD."""

import numpy as np
import matplotlib.pyplot as plt

from bci_utils import load_and_preprocess_raw, make_epochs


def main() -> None:
    # Load raw đã lọc 7–30 Hz, montage 10-20
    raw = load_and_preprocess_raw(subject=1, runs=(4, 8, 12))
    epochs, event_id = make_epochs(raw, tmin=0.0, tmax=4.0)

    # --- Thông số tổng quan ---
    print("=== RAW ===")
    print(raw)
    print(f"- Số kênh EEG: {len(raw.ch_names)}")
    print(f"- Tần số lấy mẫu: {raw.info['sfreq']} Hz")
    print(f"- Thời lượng: {raw.times[-1]:.1f} s")

    # Thông tin epochs
    X = epochs.get_data()
    labels = epochs.events[:, -1]
    print("\n=== EPOCHS ===")
    print(f"- Số trial: {len(epochs)}")
    print(f"- Shape: {X.shape} (trial, channel, sample)")
    for name, code in event_id.items():
        print(f"  + {name}: {int(np.sum(labels == code))} trial")
    print(f"- Cửa sổ thời gian: {epochs.tmin} -> {epochs.tmax} s")

    # --- Hình học: vị trí kênh ---
    fig1 = raw.plot_sensors(kind="topomap", show=False, title="Vị trí điện cực (montage)")

    # --- PSD: phổ công suất 0–50 Hz ---
    fig2 = raw.plot_psd(fmin=0, fmax=50, average=True, show=False)
    fig2.suptitle("PSD trung bình 0–50 Hz (đã lọc 7–30 Hz)", fontsize=12)

    # --- Event timeline: thời điểm cue ---
    events = epochs.events[:, 0] / raw.info["sfreq"]  # thời gian (s)
    fig3, ax3 = plt.subplots(figsize=(8, 2))
    ax3.vlines(events, ymin=0, ymax=1, color="k", alpha=0.5, linewidth=0.8)
    ax3.set_title("Thời điểm các cue (s)", fontsize=12)
    ax3.set_yticks([])
    ax3.set_xlabel("Thời gian (s)")
    ax3.set_xlim(0, raw.times[-1])

    plt.show()


if __name__ == "__main__":
    main()
