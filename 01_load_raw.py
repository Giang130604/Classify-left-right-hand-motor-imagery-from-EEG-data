"""Load EEGBCI raw data and show basic info."""

from bci_utils import load_and_preprocess_raw


def main() -> None:
    raw = load_and_preprocess_raw(subject=1, runs=(4, 8, 12))
    print(raw)
    print(f"Số kênh EEG: {len(raw.ch_names)}")
    print(f"Tần số lấy mẫu: {raw.info['sfreq']} Hz")
    print("Một vài kênh đầu:", raw.ch_names[:10])

    # Nếu muốn xem tín hiệu, bỏ comment:
    # raw.plot(n_channels=8, duration=10.0, scalings='auto', block=True)


if __name__ == "__main__":
    main()
