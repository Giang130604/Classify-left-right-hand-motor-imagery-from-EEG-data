"""Tien ich tai va tien xu ly du lieu EEGBCI (motor imagery)."""

from typing import Dict, Iterable, Optional, Tuple

import mne
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import ICA

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


def _select_eog_reference_channel(raw: mne.io.BaseRaw) -> Optional[str]:
    eog_picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=True, exclude=[])
    if len(eog_picks) > 0:
        return raw.ch_names[eog_picks[0]]
    frontal_candidates = ["FP1", "FP2", "AF7", "AF8", "AF3", "AF4", "FPZ", "AFZ"]
    name_by_upper = {ch.upper(): ch for ch in raw.ch_names}
    for candidate in frontal_candidates:
        if candidate in name_by_upper:
            return name_by_upper[candidate]
    return None


def _apply_ica_denoising(
    raw: mne.io.BaseRaw,
    filter_method: str,
    fir_design: str,
    ica_random_state: int = 97,
    ica_n_components: float | int | None = 0.99,
) -> mne.io.BaseRaw:
    raw_for_ica = raw.copy()
    raw_for_ica.filter(
        l_freq=1.0,
        h_freq=None,
        method=filter_method,
        fir_design=fir_design,
        skip_by_annotation="edge",
    )
    raw_for_ica.pick_types(meg=False, eeg=True, eog=False)
    ica = ICA(
        n_components=ica_n_components,
        method="fastica",
        random_state=ica_random_state,
        max_iter="auto",
    )
    ica.fit(raw_for_ica)
    eog_reference = _select_eog_reference_channel(raw)
    if eog_reference is not None:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_reference)
        ica.exclude = eog_indices
    cleaned = ica.apply(raw)
    return cleaned


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
    Tai va tien xu ly cac run EEGBCI cho mot subject.

    Buoc chinh:
    - Nap EDF, lam sach ten kenh, gan montage 10-20.
    - Tham chieu trung binh.
    - Fit ICA de loai nhieu EOG (su dung kenh EOG neu co, hoac Fp1/Fp2/AF).
    - Loc bang 7-30 Hz tap trung Mu/Beta phu hop MI.
    - Chi giu ROI sensorimotor, resample de dong nhat tan so.
    """
    raw_paths = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(path, preload=True, stim_channel="auto") for path in raw_paths]
    raw = concatenate_raws(raws)

    raw.rename_channels(lambda x: x.rstrip(".").upper())
    montage = make_standard_montage(montage_name)
    standard_lookup = {name.upper(): name for name in montage.ch_names}
    raw.rename_channels(lambda x: standard_lookup.get(x.upper(), x))
    raw.set_montage(montage)

    raw.set_eeg_reference("average", projection=True)

    raw = _apply_ica_denoising(raw, filter_method=filter_method, fir_design=fir_design)

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=filter_method,
        fir_design=fir_design,
        skip_by_annotation="edge",
    )

    raw.pick(picks="eeg")

    roi = list(roi_channels) if roi_channels is not None else MOTOR_ROI_CHANNELS
    roi = [ch.upper() for ch in roi]
    name_by_upper = {ch.upper(): ch for ch in raw.ch_names}
    roi_present = [name_by_upper[ch] for ch in roi if ch in name_by_upper]
    if not roi_present:
        raise ValueError(
            "Khong tim thay kenh ROI nao trong raw. "
            "Kiem tra montage/ten kenh hoac danh sach MOTOR_ROI_CHANNELS."
        )
    raw.pick(roi_present)
    raw.reorder_channels(roi_present)

    if resample_sfreq is not None:
        raw.resample(resample_sfreq)
    return raw


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = 0.5,
    tmax: float = 2.5,
    event_id_map: Dict[str, str] = None,
    reject: Optional[Dict[str, float]] = None,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Tuple[mne.Epochs, Dict[str, int]]:
    """
    Tao epochs cho left/right hand dua tren Annotations.

    - Mac dinh tmin=0.5s va tmax=2.5s bo cue som va tap trung doan imagery.
    - Mac dinh reject 120e-6 sau buoc ICA de cat artifact con lai.
    - Baseline mac dinh None vi cua so bat dau sau cue.
    """
    if event_id_map is None:
        event_id_map = {"T1": "left_hand", "T2": "right_hand"}

    if reject is None:
        reject = {"eeg": 120e-6}

    if baseline is not None:
        start, end = baseline
        if start is not None and start < tmin:
            raise ValueError("Baseline start phai nam trong khoang epoch (>= tmin).")
        if end is not None and end > tmax:
            raise ValueError("Baseline end phai nam trong khoang epoch (<= tmax).")

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
