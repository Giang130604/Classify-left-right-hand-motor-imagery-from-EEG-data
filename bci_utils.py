"""Helper functions to load, preprocess, and epoch EEGBCI motor imagery data."""

from typing import Dict, Iterable, Tuple

import mne
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


def load_and_preprocess_raw(
    subject: int = 1,
    runs: Iterable[int] = (4, 8, 12),
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    montage_name: str = "standard_1020",
) -> mne.io.BaseRaw:
    """
    Tải và tiền xử lý dữ liệu EEGBCI:
    - Tải các run EDF, ghép raw.
    - Chuẩn hóa tên kênh về chuẩn 10-20.
    - Gán montage, lọc băng 7-30 Hz.
    - Chỉ giữ kênh EEG.
    """
    raw_paths = eegbci.load_data(subject, runs)  # MNE tự tải nếu chưa có
    raws = [read_raw_edf(path, preload=True, stim_channel="auto") for path in raw_paths]
    raw = concatenate_raws(raws)

    # Chuẩn hóa tên kênh: bỏ dấu '.' và viết hoa để map sang montage
    raw.rename_channels(lambda x: x.rstrip(".").upper())
    montage = make_standard_montage(montage_name)
    # Ánh xạ tên kênh (UPPER) -> tên chuẩn trong montage (đúng chữ thường/z)
    standard_lookup = {name.upper(): name for name in montage.ch_names}
    raw.rename_channels(lambda x: standard_lookup.get(x.upper(), x))
    raw.set_montage(montage)

    # Thêm tham chiếu trung bình (projection) để giảm nhiễu chung
    raw.set_eeg_reference("average", projection=True)

    # Lọc băng để giữ Mu/Beta (liên quan vận động) và bỏ nhiễu chậm/cao
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", skip_by_annotation="edge")

    # Chỉ giữ kênh EEG
    raw.pick_types(eeg=True, eog=False, stim=False)
    return raw


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = 0.0,
    tmax: float = 4.0,
    event_id_map: Dict[str, str] = None,
) -> Tuple[mne.Epochs, Dict[str, int]]:
    """
    Tạo epochs theo cue tay trái/phải:
    - events_from_annotations để lấy T1/T2.
    - map nhãn sang tên lớp rõ ràng.
    - cắt cửa sổ tmin -> tmax.
    """
    if event_id_map is None:
        # Mặc định EEGBCI: T1 = left_hand, T2 = right_hand
        event_id_map = {"T1": "left_hand", "T2": "right_hand"}

    events, event_id = mne.events_from_annotations(raw)
    # Đổi tên event_id theo map (nếu có T1/T2)
    renamed = {}
    for key, name in event_id_map.items():
        if key in event_id:
            renamed[name] = event_id[key]

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=renamed,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )
    return epochs, renamed
