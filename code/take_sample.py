from typing import *
import keyboard
import termios
import sys
import os
from datetime import date, datetime
import config


def take_samples(sentence: str = "", n: int = 1) -> List[Tuple[str, str, List[Tuple[str, float, float]]]]:
    '''
    This function makes the user repeat n times (1 if not specified) the sentence
    sentence (can write free text if not specified) and records all the sentences written in a list of triple
    (person_name, date, events)
    '''
    person_name: str = input("Insert your name:\n")
    samples: List[Tuple[str, str, List[Tuple[str, float, float]]]] = list()
    today = str(datetime.today())
    for _ in range(n):
        samples.append(
            (person_name, today, _take_sample(sentence=sentence)))
    return samples


def _take_sample(sentence: str = "") -> List[Tuple[str, float, float]]:
    '''
    This function records the user while he writes the sentence (if not specified, free text can be written).
    It return a list of triple (keyboard_pressed, time_down, time_up)
    '''
    if sentence == "":
        print("Write something and I'll record it (press Enter when you have finished)")
    else:
        print(f"Write '{sentence}' and then press Enter")
    # until the user presses "enter", we record all the events
    records: List[keyboard.KeyboardEvent] = keyboard.record(until="enter")
    keys: List[Tuple[str, float]] = list()
    ups: List[Tuple[str, float, float]] = list()
    found = False
    # for each event
    for event in records:
        found = False
        # we search for another event with the same key pressed (that is still not matched)
        for i in range(len(ups)):
            name, timestamp = ups[i]
            # if we find a match, we remember that
            if name.lower() == event.name.lower():
                found = True
                break
        if found:
            # we store the key down and key up times
            keys.append((name, timestamp, event.time))
            ups.pop(i)
        else:
            # if there is no match, the match will come in later events
            ups.append((event.name, event.time))

    # We flush the stdin because the keyboard library would repeat the password otherwise
    if sys.platform == "linux":
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        print("\n")
    return keys


#l = take_samples(sentence="ciao", n=2)
