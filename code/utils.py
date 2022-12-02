import requests
from bs4 import BeautifulSoup
import torch
from tqdm import tqdm
import os
import pandas as pd
from typing import *
import re
import random

import config

# the regex that identifies a tuple (key,keycode,dt,wt)
#regex = r"\(\"(\w+|\s|\W|\\\w)\", (\d+|missing), \d+, \-?\d+\)"
regex = r"\(\"(\w+|\s|\W|\\\w)\", (\d+|missing), \d+(\.\d+)?, \-?\d+(.\d+)?\)"


def get_keycode_mapping() -> Tuple[Dict[int,str], Dict[str,int]]:
    '''
    JavaScript Keycode mapping obtained
    from the website www.toptal.com/developers/keycode/table-of-all-keycodes
    '''
    # get the html of the page
    url = "https://www.toptal.com/developers/keycode/table-of-all-keycodes"
    page = requests.get(url)

    js_code_to_key : Dict[int,str] = dict()
    # reverse mapping
    js_key_to_code : Dict[str,int] = dict()

    # parse the html using beautiful soup
    soup = BeautifulSoup(page.content, "html.parser")

    # look at all the rows of the table
    for row in soup.find(class_="table-body").find_all("tr"):
        # for each row, the first 2 columns contains keycode and key
        columns = row.find_all("td")
        keycode = int(columns[0].text)
        if keycode == 32: # the space is not showed on the website, we force it
            value = " "
        else:
            value = columns[1].text.lower()
        # save the mapping keycode:key
        js_code_to_key[keycode] = value
        js_key_to_code[value] = keycode
    
    return js_code_to_key, js_key_to_code


# the actual function that makes the conversion
def string_to_timings(s : str) -> List[Tuple[str,int,float,float]]:
    '''
    Given a string representing the TIMING cell of a row in the dataframe,
    this function process this string and formats that into a list of tuple,
    where each of them has:
        - key, str reperesentation of the key pressed;
        - keycode, javascript keycode of the key pressed;
        - dwell time, elapsed time between key press and key release;
        - waiting time, elapsed time between the previous key press and the actual key release.
    '''
    out : List[Tuple[str,int,float,float]] = list()
    for match in re.finditer(regex, s, re.MULTILINE):
        key, keycode, dt, wt = match.group()[1:-1].split(", ")
        key = key[1:-1] # remove " at the beginning and " at the end
        if keycode == "missing":
            keycode = key
            key = config.js_code_to_key[int(key)]
        elif key == "\\b": #TODO (?)
            key = "backspace"
        config.chars.add(key.lower())
        out.append((key.lower(),int(keycode),float(dt),float(wt)))
    return out

def string_to_vocab(s : str) -> None:
    '''
    Given a representing the TIMING cell of a row in the dataframe,
    this functions adds all the typed characters in a set called "chars"
    '''
    for match in re.finditer(regex, s, re.MULTILINE):
        key, keycode, dt, wt = match.group()[1:-1].split(", ")
        key = key[1:-1] # remove " at the beginning and " at the end
        if keycode == "missing":
            keycode = key
            key = config.js_code_to_key[int(key)]
        elif key == "\\b": #TODO (?)
            key = "backspace"
        config.chars.add(key.lower())


def string_to_tensor(s : str, vocab : Dict[str,int]) -> torch.Tensor:
    out : List[Tuple[int,float,float]] = list()
    for match in re.finditer(regex, s, re.MULTILINE):
        key, keycode, dt, wt = match.group()[1:-1].split(", ")
        key = key[1:-1] # remove " at the beginning and " at the end
        if keycode == "missing":
            keycode = key
            key = config.js_code_to_key[int(key)]
        elif key == "\\b": #TODO (?)
            key = "backspace"
        out.append((vocab.get(key.lower(),vocab[config.UNK_KEY]),float(dt),float(wt)))
    return torch.tensor(out)

def get_training_df(n : int) -> pd.DataFrame:
    '''
    This function concatenates the samples of the first n users
    into a single pandas dataframe which is then returned.
    '''
    dataframes : List[pd.DataFrame] = list()
    users : List[str] = os.listdir(config.ROOT_PATH.joinpath("data/data/Keystrokes_processed/"))
    added : int = 0
    idx : int = 0
    with tqdm(total=n) as pbar:
        while added < n:
            user = users[idx]
            if "keystrokes" in user: # it is a user file (and not readme)
                df : pd.DataFrame = pd.read_csv(config.ROOT_PATH.joinpath(f"data/data/Keystrokes_processed/{user}"),
                                            sep=",",
                                            names = config.column_names,
                                            header=None,
                                            encoding = "ISO-8859-1",
                                            )
                dataframes.append(df)
                added += 1
                pbar.update(1)
            idx += 1

    train_dataset : pd.DataFrame = pd.concat(dataframes)
    # for training we are not going to need these columns
    train_dataset = train_dataset.drop(['TEST_SECTION_ID','SENTENCE','USER_INPUT'],axis=1) 
    return train_dataset


def index_2_combination(index : int, n: int, comb_num : int = None) -> Tuple[int,int]:
    '''
    Given the number of elements n (0,1,2,...,n-1) and an index i, this function returns
    the index-th combinations with length 2 without repetition.
    The combinations are ordered in ascending order.
    '''
    if comb_num is None: # if not given, compute the number of possible couples
        comb_num : int = n*(n-1)//2 # the number of combinations
    cur_index = index + 1
    assert(cur_index <= comb_num) # assert you do not go out of index
    first_e : int = 0 # the first element of the combination -> (0,?)
    # while you can subtract by (n-e) it means that the index refers to a number > e
    while cur_index - n + first_e + 1 > 0: 
        cur_index -= n - (first_e + 1)
        first_e += 1
    # the remainder gives us the second element
    return first_e, (first_e + cur_index)


def train_val_split(df : pd.DataFrame, perc : float):
    '''
    This function splits the pandas dataframe into two
    dataframes, giving to the first (1-perc)% of the strings
    '''
    # all the users of our dataset
    users: Set[str] = sorted(set(df.PARTICIPANT_ID))

    # the number of users we will keep in the training set
    train_num = int(len(users)*(1-perc))

    # sample the users at random
    training_users: Set[str] = random.sample(users, k=train_num)

    # the boolean series for the rows of the training dataframe
    train_series: pd.Series = df.PARTICIPANT_ID.isin(training_users)

    # the validation set is taken from the users not kept in the training set
    val_dataset: pd.DataFrame = df[~train_series]

    # the remaining users will be in our training set
    train_dataset = df[train_series]

    return train_dataset, val_dataset 


def compute_vocab(df : pd.DataFrame) -> None:
    '''
    This functions computes the character vocabulary based on
    the given dataframe
    '''
    config.chars = set()
    df.TIMINGS.apply(string_to_vocab)
    # compute vocabulary only for the training set
    config.char_vocab : Dict[str,int] = { c:i+2 for i,c in enumerate(sorted(config.chars)) }
    config.char_vocab[config.PAD_KEY] = 0
    config.char_vocab[config.UNK_KEY] = 1