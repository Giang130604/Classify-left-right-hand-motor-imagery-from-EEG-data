"""Các hàm tiện ích để nạp và tiền xử lý dữ liệu EEGBCI (motor imagery)."""

from typing import Dict, Iterable, Optional, Tuple

import mne
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


# Các kênh ROI nằm trên vùng vỏ não vận động (Sensorimotor cortex)
MOTOR_ROI_CHANNELS = [
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
]


def load_and_preprocess_raw(
    subject: int = 1,
    runs: Iterable[int] = (4, 8, 12),
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    montage_name: str = "standard_1020",
    resample_sfreq: Optional[float] = 160.0,
    filter_method: str = "fir",
    fir_design: str = "firwin",
    roi_channels: Optional[Iterable[str]] = None,
) -> mne.io.BaseRaw:
    """
    Tải (nếu cần) và tiền xử lý các run EEGBCI cho 1 subject.

    Các bước chính:
    - Nạp EDF, chuẩn hoá tên kênh, gán montage chuẩn.
    - Tham chiếu trung bình (average reference).
    - Lọc thông dải FIR 7–30 Hz (Mu/Beta liên quan vận động).
    - Chỉ giữ EEG và chọn ROI vùng vỏ não vận động (giảm nhiễu vùng trán/mắt).
    - Resample về cùng tần số lấy mẫu để tránh lệch độ dài epoch giữa subject.
    """
    raw_paths = eegbci.load_data(subject, runs)  # MNE auto-downloads to MNE_DATA
    raws = [read_raw_edf(path, preload=True, stim_channel="auto") for path in raw_paths]
    raw = concatenate_raws(raws)

    # Clean channel names and set montage
    raw.rename_channels(lambda x: x.rstrip(".").upper())
    montage = make_standard_montage(montage_name)
    standard_lookup = {name.upper(): name for name in montage.ch_names}
    raw.rename_channels(lambda x: standard_lookup.get(x.upper(), x))
    raw.set_montage(montage)

    # Average reference projection
    raw.set_eeg_reference("average", projection=True)

    # FIR band-pass 7-30 Hz (default)
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=filter_method,
        fir_design=fir_design,
        skip_by_annotation="edge",
    )

    # Chỉ giữ EEG (inst.pick là API mới)
    raw.pick(picks="eeg")

    # Chọn ROI vùng vỏ não vận động để tập trung tín hiệu Mu/Beta
    roi = list(roi_channels) if roi_channels is not None else MOTOR_ROI_CHANNELS
    roi = [ch.upper() for ch in roi]
    name_by_upper = {ch.upper(): ch for ch in raw.ch_names}
    roi_present = [name_by_upper[ch] for ch in roi if ch in name_by_upper]
    if not roi_present:
        raise ValueError(
            "Không tìm thấy kênh ROI nào trong raw. "
            "Hãy kiểm tra montage/tên kênh hoặc danh sách MOTOR_ROI_CHANNELS."
        )
    # inst.pick là API mới; reorder để giữ đúng thứ tự ROI đã khai báo
    raw.pick(roi_present)
    raw.reorder_channels(roi_present)

    # Resample so all subjects share the same sampling rate (avoids length mismatches)
    if resample_sfreq is not None:
        raw.resample(resample_sfreq)
    return raw


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = -0.5,
    tmax: float = 3.5,
    event_id_map: Dict[str, str] = None,
    reject: Optional[Dict[str, float]] = None,
    baseline: Tuple[Optional[float], Optional[float]] | None = (None, 0.0),
) -> Tuple[mne.Epochs, Dict[str, int]]:
    """
    Tạo epochs cho left/right hand dựa trên Annotations.

    - Artifact rejection: loại epoch nếu biên độ EEG vượt 100 µV (thường do chớp mắt/nghiến răng).
    - Baseline correction: baseline=(None, 0) để đưa tín hiệu về mức chuẩn trước kích thích.
    """
    if event_id_map is None:
        event_id_map = {"T1": "left_hand", "T2": "right_hand"}

    # Nới ngưỡng reject mặc định lên 180 µV để tránh loại sạch epoch khi dữ liệu còn thô
    if reject is None:
        reject = {"eeg": 180e-6}

    if baseline is not None and baseline[1] is not None and tmin > baseline[1]:
        raise ValueError(
            "Baseline (None, 0) yêu cầu epoch phải chứa mốc thời gian 0s. "
            "Hãy đặt tmin <= 0 (ví dụ tmin=-0.5)."
        )

    events, event_id = mne.events_from_annotations(raw)
    renamed: Dict[str, int] = {}
    for key, name in event_id_map.items():
        if key in event_id:
            renamed[name] = event_id[key]

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=renamed,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        reject_by_annotation=True,
    )
    return epochs, renamed
