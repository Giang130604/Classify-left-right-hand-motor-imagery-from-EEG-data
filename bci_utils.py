"""Helper functions to load, preprocess, and epoch EEGBCI motor imagery data."""

from typing import Dict, Iterable, Optional, Tuple

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
    resample_sfreq: Optional[float] = 160.0,
    filter_method: str = "fir",
    fir_design: str = "firwin",
) -> mne.io.BaseRaw:
    """
    Download (if needed) and preprocess EEGBCI runs for one subject.
    Steps: load EDFs, rename channels, set montage, average reference,
    FIR band-pass 7-30 Hz, keep EEG, optional resample for consistent epoch length.
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

    # Keep EEG only (inst.pick is the modern API)
    raw.pick(picks="eeg")

    # Resample so all subjects share the same sampling rate (avoids length mismatches)
    if resample_sfreq is not None:
        raw.resample(resample_sfreq)
    return raw


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = 0.5,
    tmax: float = 3.5,
    event_id_map: Dict[str, str] = None,
) -> Tuple[mne.Epochs, Dict[str, int]]:
    """Create epochs for left/right hand cues using event annotations."""
    if event_id_map is None:
        event_id_map = {"T1": "left_hand", "T2": "right_hand"}

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
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )
    return epochs, renamed
