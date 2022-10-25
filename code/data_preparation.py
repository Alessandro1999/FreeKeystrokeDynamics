from typing import *
from pathlib import Path
import re
import os
import pandas as pd
import torch
import random

import config
import old_dataset_conversion as old
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


def compute_metrics(sample: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    '''
    Given the timings of a recorded sentence, it computes for each key:
    - the dwell time, the time between a key press and its release
    - waiting time, time between the previous key is released and the actual is pressed

    Other metrics can be computed but are ignored since are a combination of the two above:
    - releasing interval, the time between the previous key is released and the actual one is released (NOT USED since it is basically dwell time + waiting time)
    - pressing interval, time between pressing of the previous key and pressing of the current one (NOT USED since it is prev dwell time + waiting time)
    - double typing time, time between the pressing of the previous key and the releasing of the current one (NOT USED since it is dwell time + prev dwell time + waiting time)

    '''
    output = list()
    for i, (key, press_time, release_time) in enumerate(sample):
        key = key.lower()
        if "key" in key:
            key = key.split(":")[0]
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

    return output


def pd_conversion(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Subject", "Date", "Sentence", "Timings"]
    data = list()
    for i in range(df.shape[0]):
        # subject,date,sentence
        s = df.iloc[i][0]
        row = [(s if s in config.known_subject else config.UNK_SUB),
               df.iloc[i][1], df.iloc[i][2]]
        timings = compute_metrics(string_to_seq(df.iloc[i][3]))
        row.append(timings)
        data.append(row)

    return pd.DataFrame(data=data, columns=columns)


def get_dataframes(path: Path) -> pd.DataFrame:
    '''
    Given a path, all the files in the path (that shuold be .csv)
    are transformed into a dataframe and concatenated (this means that all these dataframes
    should have the same columns)
    '''
    dfs: List[pd.DataFrame] = list()
    for file in os.listdir(path):  # for each file in the path
        # read the dataframe
        tdf: pd.DataFrame = pd.read_csv(path.joinpath(file))
        # if the file was an old dataset (different columns)
        if "[OLD]" in file:
            # convert the file to the new format
            cdf = old.pd_conversion(tdf)
        else:
            # if it is not old, do the standard conversion
            cdf = pd_conversion(tdf)
        dfs.append(cdf)

    if len(dfs) == 0:
        return None
    # return the dataframes concatenated
    return pd.concat(dfs)


def split_df(df: pd.DataFrame, perc: float) -> Tuple[pd.DataFrame]:
    '''
    Given a pandas dataframe and a percentage, the dataframe is split
    in two new dataframes where the first one has the given percentage
    of rows and the second 1-percentage; the intersection between them is
    empty and they are splitted according to the Date in which the samples are
    taken
    '''
    dates: Set[str] = sorted(set(df.Date))  # all the dates in the dataframe

    # the number of dates to assign to the first dataframe
    df1_num = int(len(dates)*perc)
    # pick randomly df1_num of dates
    df1_dates: Set[str] = random.sample(dates, k=df1_num)

    # the boolean series for the rows of the first dataframe
    df1_series: pd.Series = df.Date.isin(df1_dates)

    # the first dataframe picks the rows that have the dates selected
    df1: pd.DataFrame = df[df1_series]
    # the second dataframe picks all the others (~ is a logical pointwise not)
    df2: pd.DataFrame = df[~df1_series]
    return df1, df2


def get_keys(df: pd.DataFrame) -> Set[str]:
    '''
    Given a pandas dataframe, it returns the set of all the keystroke pressed
    in all the samples of the dataframe
    '''
    # each row contains its own set of keystroke pressed
    s: pd.Series = df.Timings.apply(
        lambda event: {key for key, dt, wt in event})
    keys = set()
    # apply the set union of each row
    s.aggregate(lambda x: keys.update(x))
    return keys


# TODO unused
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
