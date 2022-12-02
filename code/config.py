from typing import *
from pathlib import Path
import sys

# define the path of the root folder indipendently from where the code is executed
ROOT_PATH: Path = Path(__file__).parent.parent

# the columns of our dataframes
column_names = ['PARTICIPANT_ID','TEST_SECTION_ID','SENTENCE','USER_INPUT','TIMINGS']

# append the folder to the path
sys.path.append(str(ROOT_PATH))

# unknown key
UNK_KEY = "<unk>"
# pad key
PAD_KEY = "<pad>"

# the seed for random stuff
seed: int = 17

# mapping from javascript keycodes to the respective key
js_code_to_key : Dict[int,str] = dict()

# inverse mapping
js_key_to_code : Dict[str,int] = dict()

# this set will contain all the characters used in typing (for the training set)
chars : Set[str] = set()