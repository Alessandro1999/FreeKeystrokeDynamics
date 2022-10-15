from typing import *
import os
import sys
import time
import re
from datetime import datetime
import random
from pathlib import Path
import pandas as pd
from pynput import keyboard

regex: str = r"\(((\<\S+\: ((\' \')|\<\d+\>)\>)|(\'\w\'))\, \d+\.\d+\, \d+\.\d+\)"

italian_pangrams: List[str] = ["Che tempi brevi zio, quando solfeggi",
                               "Pranzo d'acqua fa volti sghembi",
                               "Qualche vago ione tipo zolfo, bromo, sodio",
                               "O templi, quarzi, vigne, fidi boschi!",
                               "Ma che bel gufo spenzola da quei travi"]


def flush_stdin() -> None:
    # We flush the stdin because the keyboard library would repeat the password otherwise
    if sys.platform == "linux":  # on linux, we use the termios library
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        print("\n")
    elif sys.platform == "win32" or sys.platform == "win64":  # on windows, the msvrct
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()


def record(sentence: str = "") -> Tuple[List[Tuple[str, float, float]], str]:
    # The event listener will be running in this block
    header: str = "Type something (press Enter to finish)" if sentence == "" else f"Type '{sentence}' (press Enter to finish)"
    s = ""  # this variable will contain what the user writes
    # this variable tells if we interpret charachters as lower or upper case
    upper: bool = False
    # pending keys waiting to be released
    pending: List[Tuple[keyboard.KeyCode, float]] = list()
    output: List[Tuple[keyboard.KeyCode, float, float]] = list()
    print(header)
    print("|")
    with keyboard.Events() as events:
        for event in events:  # wait for an event
            if event.key == keyboard.Key.enter:  # if the user presses enter, get out
                break

            # keys that can change the uppercase to lowercase and viceversa
            upper_changer: Set[keyboard.Key] = {keyboard.Key.shift,
                                                keyboard.Key.caps_lock,
                                                keyboard.Key.shift_l,
                                                keyboard.Key.shift_r}

            t = time.time()  # the timestamp
            if isinstance(event, keyboard.Events.Press):  # its a press event
                pending.append((event.key, t))
            elif isinstance(event, keyboard.Events.Release):  # its a release event
                # find the corresponding press event
                for i in range(len(pending)):
                    if pending[i][0] == event.key:  # found it
                        output.append((event.key, pending[i][1], t))
                        pending.pop(i)  # remove it from the pending ones
                        break

            # the key is a write key
            if event.key not in upper_changer:
                if isinstance(event, keyboard.Events.Release):
                    if sys.platform == "linux":
                        os.system("clear")
                    elif sys.platform == "win32" or sys.platform == "win64":
                        os.system("cls")
                    if event.key == keyboard.Key.backspace:
                        if len(s) >= 1:
                            s = s[:-1]
                    elif event.key == keyboard.Key.space:
                        s = s + " "
                    else:
                        char = str(event.key)[1:-1]
                        s = s + (char if not upper else char.upper())
                    print(header)
                    print(s+"|")
            elif event.key != keyboard.Key.caps_lock or not isinstance(event, keyboard.Events.Release):
                upper = not(upper)

    flush_stdin()

    if sentence != "" and s != sentence:  # you have written the wrong sentence
        raise Exception(f"You have written '{s}' instead of '{sentence}'")
    return output, s


def _take_sample(free: bool = False) -> Tuple[str, List[Tuple[str, float, float]], str]:
    sentence = ""
    if not free:
        sentence = random.choice(italian_pangrams)
    timings, s = record(sentence=sentence)
    return datetime.today().strftime("%d-%m-%Y"), timings, s


def take_sample(n: int = 1, free: bool = False, df_path: Path = None) -> pd.DataFrame:
    name = input("Who are you?\n")
    data = list()
    for _ in range(n):
        time.sleep(0.5)
        date, timings, sentence = _take_sample(free=free)
        # we sort the keys by the time they are pressed
        timings.sort(key=lambda x: x[1])
        data.append((name, date, sentence, timings))

    columns = ["Subject", "Date", "Sentence", "Timings"]
    df: pd.DataFrame = pd.DataFrame(data=data, columns=columns)

    if df_path is not None:
        old_df = pd.read_csv(df_path)
        print(old_df)
        print(df)
        df = pd.concat([old_df, df])
    return df  # TODO regex to collect the data


def string_to_seq(input: str) -> List[Tuple[str, float]]:
    matches = re.finditer(regex, input, re.MULTILINE)
    out = []
    for match in matches:
        key, press_time, release_time = match.group()[1:-1].split(",")
        # float(press_time), float(release_time)))
        out.append(
            (key[1:-1], round(float(release_time) - float(press_time), 2)))
    return out
