import pathlib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import pathlib as pl
import getopt
import sys
import re
import multiprocessing as mp

from mne.io.edf.edf import RawEDF
from dataclasses import dataclass

Idx = int
SubjectList = list[pl.Path]

"""
shape[0] -> Subjects
shape[1] -> Runs
shape[2] -> Epochs
"""
ParsedSub = dict[pathlib.Path, pd.DataFrame]

"""
Train / Val set = 80 % of the dataset
    Train set   = 80 % out of the 80% = 64%
    Val set     = 20 % out of the 80% = 16%
Test set        = 20 % of the dataset
"""
TEST_SET_SIZE = 0.2
VAL_SET_SIZE = 0.2
TRAIN_SET_SIZE = 0.8

# Band of time we keep for the data-set
LOW_TIME = 0.5
HIGH_TIME = 4

MAX_SEQUENCE = 29
SIG_POINT_NB = 561

def shuffle_subjects(subjects: pl.Path) -> tuple[SubjectList, SubjectList, SubjectList]:
    """
    Function to shuffle subjects into train, val and test sets.

    :param subjects: The path to the subjects folder.
    :return: Train, Val and Test sets.
    """

    assert subjects.exists() and subjects.is_dir(), "The main directory must contain a 'files' subdirectory."

    # Get subject list
    subjects_list = np.array(list(subjects.iterdir()))
    sub_idx = np.arange(subjects_list.shape[0])

    # Shuffle the index list
    np.random.shuffle(sub_idx)

    train_idx = sub_idx[: (sub_idx.shape[0] * TRAIN_SET_SIZE).__ceil__()]
    test_idx = sub_idx[-(sub_idx.shape[0] * TEST_SET_SIZE).__floor__():]

    val_idx = train_idx[-(train_idx.shape[0] * VAL_SET_SIZE).__floor__():]
    train_idx = train_idx[:(train_idx.shape[0] * TRAIN_SET_SIZE).__ceil__()]

    return subjects_list[train_idx].tolist(), subjects_list[val_idx].tolist(), subjects_list[test_idx].tolist()


def extract_edf_data_with_epochs(raw: RawEDF, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract EDF data, cut them in the different sequences (T0/T1/T2) and limit there size to a specific range.

    :param raw:     Raw EDF file.
    :param events:  The sequences informations.
    :return: X and y data of raw.
    """

    event_id = {'T0': 1, 'T1': 2, 'T2': 3}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=LOW_TIME,
        tmax=HIGH_TIME,
        baseline=None,
        preload=True,
        verbose=False
    )

    X_extract = epochs.get_data(copy=True)
    y_extract = epochs.events[:, -1] - 1

    if y_extract.shape[0] > MAX_SEQUENCE:
        X_extract = X_extract[:MAX_SEQUENCE, :, :]
        y_extract = y_extract[:MAX_SEQUENCE]
    elif y_extract.shape[0] < MAX_SEQUENCE:
        raise ValueError("The number of sequence is too small")

    if X_extract.shape[2] > SIG_POINT_NB:
        X_extract = X_extract[:, :, :SIG_POINT_NB]
    elif X_extract.shape[2] < SIG_POINT_NB:
        raise ValueError("The number of point in the signal is too small")

    return X_extract, y_extract


def extraxt_edf_data(raw: RawEDF) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to extract EDF data from raw EDF file.

    :param raw: Raw EDF file.
    :return:    Extracted EDF data (X and y).
    """

    raw_events, event_labels = mne.events_from_annotations(raw, verbose=False)

    """
    Extracted:
    [
        [Label (T0 / T1 / T2), [Point of the signal]]
    ]
    """
    X_extract, y_extract = extract_edf_data_with_epochs(raw, raw_events)

    return X_extract, y_extract


def __parse_pool_subject(args) -> None:
    """
    Function to parse and save a subject with it's `.edf` files to a `.npz` file.
    :param args: The path to the subjects folder and the path to save the folder.
    :return:
    """

    sub_path, save_path = args

    X_unilateral = []
    y_unilateral = []
    X_bilateral = []
    y_bilateral = []

    i = 0

    for run in sorted(sub_path.glob("*.edf")):
        if re.search("R0[12]$", run.stem):
            continue
        try:
            data = mne.io.read_raw_edf(run, verbose=False)
            X_extract, y_extract = extraxt_edf_data(data)

            if re.search(r"R(03|04|07|08|11|12)\.edf$", run.name):
                X_unilateral.append(X_extract)
                y_unilateral.append(y_extract)
            elif re.search(r"R(05|06|09|10|13|14)\.edf$", run.name):
                X_bilateral.append(X_extract)
                y_bilateral.append(y_extract)
        except Exception as e:
            print(f"Failed to read {run.name}: {e}")
            continue

    if len(X_unilateral) > 0:
        X_uni = np.concatenate(X_unilateral, axis=0)
        y_uni = np.concatenate(y_unilateral, axis=0)
        uni_dir = save_path / "unilateral"
        uni_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(uni_dir / f"{sub_path.name}.npz", X=X_uni, y=y_uni)

    if len(X_bilateral) > 0:
        X_bi = np.concatenate(X_bilateral, axis=0)
        y_bi = np.concatenate(y_bilateral, axis=0)
        bi_dir = save_path / "bilateral"
        bi_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bi_dir / f"{sub_path.name}.npz", X=X_bi, y=y_bi)


def parse_and_save_sub(sub_path_list: SubjectList, save_path: pathlib.Path) -> None:
    """
    Function to extract subject data from raw EDF file.

    :param save_path:       Path to save extracted subject data.
    :param sub_path_list:   List of subject paths.
    :return:
    """

    cpu_count = (mp.cpu_count() * 0.8).__ceil__()

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with mp.Pool(cpu_count) as pool:
        res = pool.map_async(
            __parse_pool_subject,
            [(sub, save_path) for sub in sub_path_list])

        res.get()
        pool.close()
        pool.join()


def fuse_files(save_path: pathlib.Path) -> None:
    """
    Function that create a `.npz` file which concatenate all the parsed subjects.

    :param save_path: Path to save the .npz file.
    :return:
    """

    states = ("unilateral", "bilateral")

    for state in states:
        curr_path = save_path / state
        if not curr_path.exists():
            continue

        X_all = []
        y_all = []

        for path in curr_path.glob("*.npz"):
            curr_data = np.load(path)
            X_all.append(curr_data["X"])
            y_all.append(curr_data["y"])

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        np.savez_compressed(f"{curr_path}.npz", X_all=X_all, y_all=y_all)


def main():
    """
    This scrypt is to split, specifically the EEGMMIDB data set.

    :return:
    """

    try:
        main_dir = pl.Path("./eeg-motor-movementimagery-dataset-1.0.0")
        save_dir = pl.Path("./data")

        if not main_dir.exists():
            raise FileNotFoundError("""EEGMMIDB data set not found.
            If not available, download and extract it from this website:
            https://physionet.org/content/eegmmidb/1.0.0/#files-panel""")

        save_dir.mkdir(parents=True, exist_ok=True)

        train_sub, val_sub, test_sub = shuffle_subjects(main_dir / "files")

        train_path = save_dir / "train"
        val_path = save_dir / "val"
        test_path = save_dir / "test"

        print("Parse and save training data...")
        parse_and_save_sub(train_sub, train_path)
        fuse_files(train_path)
        print("Parse and save validation data...")
        parse_and_save_sub(val_sub, val_path)
        fuse_files(val_path)
        print("Parse and save testing data...")
        parse_and_save_sub(test_sub, test_path)
        fuse_files(test_path)

    except Exception as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    main()
