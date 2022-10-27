from typing import *
import os
import sys
import time
from datetime import datetime
import random
from pathlib import Path
import pandas as pd
from pynput import keyboard

italian_sentences: List[str] = ["Che tempi brevi zio, quando solfeggi.",
                                "Pranzo d'acqua fa volti sghembi.",
                                "Qualche vago ione tipo zolfo, bromo, sodio.",
                                "O templi, quarzi, vigne, fidi boschi!",
                                "Ma che bel gufo spenzola da quei travi.",  # up to here, they are pangrams
                                "Chi conosce tutte le risposte, non si è fatto tutte le domande.",
                                "Nel tempo dell'inganno universale, dire la verità è un atto rivoluzionario.",
                                "Il bisogno di avere ragione è il segno di una mente volgare.",
                                "Tutto è difficile prima di essere semplice.",
                                "Chi disprezza la gloria otterrà quella vera."
                                ]


def flush_stdin() -> None:
    '''
    Function to flush the std input
    '''
    # We flush the stdin because the keyboard library would repeat the password otherwise
    if sys.platform == "win32" or sys.platform == "win64":  # on windows, the msvrct library is used
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    else:  # on linux, we use the termios library
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        print("\n")


def record(sentence: str = "", sample_number: str = "") -> Tuple[List[Tuple[str, float, float]], str]:
    '''
    Given a sentence (if not given the user can write free text), this function records
    the sequence and time informations of all the keys pressed and returns:
    - A list of triple, where each triple is an event (key, time pressed, time released)
    - The sentence the user has written 
    '''
    # The event listener will be running in this block
    header: str = f"({sample_number})Type something (press Enter to finish)" if sentence == "" else f"({sample_number})Type '{sentence}' (press Enter to finish)"
    s = ""  # this variable will contain what the user writes
    # this variable tells if we interpret charachters as lower or upper case
    caps_lock: bool = False
    # pending keys waiting to be released
    pending: List[Tuple[keyboard.KeyCode, float]] = list()
    # final output list
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
                # we add it to the list of pending events (waiting for the release event)
                pending.append((event.key, t))
            elif isinstance(event, keyboard.Events.Release):  # its a release event
                # find the corresponding press event
                for i in range(len(pending)):
                    if str(pending[i][0]).lower() == str(event.key).lower():  # found it
                        # add it to the final output
                        output.append((pending[i][0], pending[i][1], t))
                        pending.pop(i)  # remove it from the pending ones
                        break

            # the key is a write key
            if event.key not in upper_changer:
                if isinstance(event, keyboard.Events.Release):
                    # clear the std output (two different ways in unix or windows)
                    if sys.platform == "win32" or sys.platform == "win64":
                        os.system("cls")
                    else:
                        os.system("clear")
                    if event.key == keyboard.Key.backspace:  # with the backspace, delete the last character
                        if len(s) >= 1:
                            s = s[:-1]
                    elif event.key == keyboard.Key.space:  # with the space, add a space
                        s = s + " "
                    else:  # with a normal character, simply add it
                        char = str(output[-1][0])[1:-1]
                        s = s + (char.lower() if char.islower()
                                 != caps_lock else char.upper())
                    print(header)
                    print(s+"|")
            # the user pressed the caps lock that changes the upper to lowercase and viceversa
            elif event.key == keyboard.Key.caps_lock and isinstance(event, keyboard.Events.Press):
                caps_lock = not (caps_lock)

    flush_stdin()

    if sentence != "" and s != sentence:  # you have written the wrong sentence
        raise Exception(f"You have written '{s}' instead of '{sentence}'")
    return output, s


def _take_sample(free: bool = False, sample_number: str = "") -> Tuple[str, List[Tuple[str, float, float]], str]:
    '''
    This function takes a single sample (if free is True, from free text, otherwise between a list of
    pre-selected sentences)
    Returns:
    - today's date
    - the recordings
    - the string written
    '''
    sentence = ""
    if not free:
        sentence = random.choice(italian_sentences)
    timings, s = record(sentence=sentence, sample_number=sample_number)
    return datetime.today().strftime("%d-%m-%Y"), timings, s


def take_sample(n: int = 1, free: bool = False, df_path: Path = None, save_to_file: bool = True) -> pd.DataFrame:
    '''
    This function aquires n samples (if free is True, from free text). If df_path is specified,
    the samples will be added to the pandas dataframe found in df_path
    (if no pandas dataframe is found, a new one will be created).
    If df_path is not given, it will be considered as subject_name.csv
    Returns:
    - the pandas dataframe with the new records appended
    '''
    name = input("Who are you?\n")  # get subject name
    data = list()
    i = 0
    while i < n:  # for n (successfull) times
        time.sleep(0.5)
        try:
            date, timings, sentence = _take_sample(
                free=free, sample_number=i+1)  # get a sample
            # we sort the keys by the time they are pressed
            timings.sort(key=lambda x: x[1])
            # append the sample to the data
            data.append((name, date, sentence, timings))
            i += 1
        except:  # the sample was not acquired correctly
            print("You have written the wrong sentence, try again")

    columns = ["Subject", "Date", "Sentence", "Timings"]
    # create a dataframe for the acquired sample
    df: pd.DataFrame = pd.DataFrame(data=data, columns=columns)

    # if the path is not given, get it from the subject name
    if df_path is None:
        df_path = f"{name}.csv"

    if save_to_file:
        try:  # try to load the pandas dataframe in df_path
            old_df = pd.read_csv(df_path)
            # if you can, concatenate it with the new data
            df = pd.concat([old_df, df])
        except:  # if you can't, the dataframe will just be the new data
            pass
        finally:  # finally, save the dataframe to df_path and return it
            df.to_csv(df_path, index=False)

    return df
