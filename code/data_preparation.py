from typing import *
from pathlib import Path
import re
import os
import pandas as pd
import torch

import config
import dataset

# regex to split the string of the records in the list of events
regex: str = r"\(((\<\S+\: ((\' \')|\<\d+\>)\>)|(\'\S\'|\"\'\"))\, \d+\.\d+\, \d+\.\d+\)"


def string_to_seq(input: str) -> List[Tuple[str, float, float]]:
    '''
    Given an input string representing the recordings of a sentence,
    this function returns it as a list of key,Dwell time for each key pressed
    '''
    matches = re.finditer(regex, input, re.MULTILINE)
    out = []
    for match in matches:
        key, press_time, release_time = match.group()[1:-1].split(", ")
        out.append((key[1:-1],
                    round(float(press_time), 5),
                    round(float(release_time), 5)))
    return out


def compute_metrics(sample: List[Tuple[str, float, float]]) -> Tuple[List[Tuple[float, float]], Set[str]]:
    '''
    Given the timings of a recorded sentence, it computes for each key:
    - the dwell time, the time between a key press and its release
    - waiting time, time between the previous key is released and the actual is pressed

    Other metrics can be computed but are ignored since are a combination of the two above:
    - releasing interval, the time between the previous key is released and the actual one is released (NOT USED since it is basically dwell time + waiting time)
    - pressing interval, time between pressing of the previous key and pressing of the current one (NOT USED since it is prev dwell time + waiting time)
    - double typing time, time between the pressing of the previous key and the releasing of the current one (NOT USED since it is dwell time + prev dwell time + waiting time)

    Also, it returns the set of the keys pressed by the user ins the recorded sentence
    '''
    output = list()
    keys = set()
    for i, (key, press_time, release_time) in enumerate(sample):
        key = key.lower()
        if "key" in key:
            key = key.split(":")[0]
        keys.add(key)
        dwell_time = round(release_time - press_time, 5)
        if i == 0:
            waiting_time = 0
            # releasing_interval = 0
            # pressing_interval = 0
            # double_typing_time = 0
        else:
            _, prev_press_time, prev_release_time = sample[i-1]
            waiting_time = round(press_time - prev_release_time, 5)
            # releasing_interval = release_time - prev_release_time
            # pressing_interval = press_time - prev_press_time
            # double_typing_time = release_time - prev_press_time
        output.append((key,
                       dwell_time,
                       waiting_time,))

    return output, keys


def pd_to_lists(df: pd.DataFrame) -> Tuple[List[List[Tuple[str, float, float]]], Set[str]]:
    '''
    Given a pandas dataframe, it returns the timing metrics along with the set of all keys pressed
    '''
    out = list()
    all_keys = set()
    for i in range(df.shape[0]):
        metrics, keys = compute_metrics(string_to_seq(df.Timings.iloc[i]))
        all_keys = all_keys.union(keys)
        out.append(metrics)
    return out, all_keys


def pd_to_dataset(df: pd.DataFrame, compute_vocab: bool = True) -> dataset.KeystrokeDataset:
    data, keys = pd_to_lists(df)

    if compute_vocab:
        subjects: Set[str] = set(df.Subject)
        config.subject_map: Dict[str, int] = {
            s: i+1 for i, s in enumerate(sorted(subjects))}
        config.subject_map[config.UNK_SUB] = 0

        config.key_map: Dict[str, int] = {
            k: i+2 for i, k in enumerate(sorted(keys))}
        config.key_map[config.PAD_KEY] = 0
        config.key_map[config.UNK_KEY] = 1

    ground_truth = torch.zeros(len(data))
    lengths = torch.zeros(len(data))
    keys_list = list()
    time_list = list()
    subjects = list(df.Subject)
    for i in range(len(data)):
        ground_truth[i] = config.subject_map[subjects[i]]
        lengths[i] = len(data[i])
        kt = torch.zeros(len(data[i]), dtype=torch.long)
        t = torch.zeros(len(data[i]), 2)
        for j, (k, press, release) in enumerate(data[i]):
            kt[j] = config.key_map[k]
            t[j][0] = press
            t[j][1] = release
        keys_list.append(kt)
        time_list.append(t)

    keys_tensor = torch.nn.utils.rnn.pad_sequence(
        keys_list, batch_first=True, padding_value=config.key_map[config.PAD_KEY])

    time_tensor = torch.nn.utils.rnn.pad_sequence(
        time_list, batch_first=True, padding_value=0.0)

    return dataset.KeystrokeDataset(ground_truth, keys_tensor, time_tensor, lengths)


def get_dataframes(path: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = list()
    for file in os.listdir(path):
        tdf: pd.DataFrame = pd.read_csv(path.joinpath(file))
        dfs.append(tdf)

    if len(dfs) == 0:
        return None
    return pd.concat(dfs)
