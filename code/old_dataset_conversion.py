from typing import *
import pandas as pd
import config


def row_conversion(row: List[float]) -> List[Tuple[str, float, float]]:
    new_row = [('p', row[2], 0)]
    row = row[3:]
    missing_letters = "assword4592,i4"
    timings = None
    for v in range(len(row)):
        r = v % 5
        if r == 0:
            l = missing_letters[0]
            missing_letters = missing_letters[1:]
            timings = (l, row[v])
        elif r == 1:
            timings = (timings[0], timings[1], row[v])
            new_row.append(timings)
    return new_row


def pd_conversion(df: pd.DataFrame) -> Tuple[List[List[Tuple[str, float, float]]], Set[str]]:
    columns = ["Subject", "Date", "Sentence", "Timings"]
    sentence = "password4592,i4"
    data = list()
    for i in range(df.shape[0]):
        s = df.iloc[i][0]
        row = [s if s in config.known_subject else config.UNK_SUB]  # subject
        year, month, day = df.iloc[i][1].split("-")
        row.append(f"{day}-{month}-{year}")  # date
        row.append(sentence)  # sentence
        row.append(row_conversion(list(df.iloc[i])))  # timings
        data.append(row)
        # data.append(row_conversion(list(df.iloc[i])))

    new_df = pd.DataFrame(data=data, columns=columns)
    return new_df, set("password4592,i4")
