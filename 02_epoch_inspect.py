"""Create epochs and inspect their shape/event counts."""

import numpy as np

from bci_utils import load_and_preprocess_raw, make_epochs


def main() -> None:
    raw = load_and_preprocess_raw(subject=1, runs=(4, 8, 12))
    epochs, event_id = make_epochs(raw, tmin=0.0, tmax=4.0)

    print(f"Số trial: {len(epochs)}; event_id: {event_id}")
    print(f"Shape epochs: {epochs.get_data().shape} (trials, channels, samples)")

    # Đếm số trial theo lớp
    labels = epochs.events[:, -1]
    for name, code in event_id.items():
        count = int(np.sum(labels == code))
        print(f"- {name}: {count} trial")

    # Nếu muốn xem trung bình theo lớp, bỏ comment:
    # epochs['left_hand'].average().plot()
    # epochs['right_hand'].average().plot()


if __name__ == "__main__":
    main()
