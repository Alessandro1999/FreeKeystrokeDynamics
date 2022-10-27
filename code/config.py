from typing import *
from pathlib import Path
import sys

# define the path of the root folder indipendently from where the code is executed
ROOT_PATH: Path = Path(__file__).parent.parent

# append the folder to the path
sys.path.append(str(ROOT_PATH))

# unknown subject
UNK_SUB = "<UNK>"

# unknown key
UNK_KEY = "<unk>"
# pad key
PAD_KEY = "<pad>"

# mapping from a key to a one-hot value
key_map: Dict[str, int] = dict()

# mapping from a subject name to a one-hot value
subject_map: Dict[str, int] = dict()

# non-unknown subject
known_subject: Set[str] = {"Alessandro", "Palo", "Iolanda", "Paglialunga",
                           "Helena", "Leonardo", "Bianca", "Roberto", "RobertoM", "Umberto", "AlessandroPecchini"}
#known_subject: Set[str] = {"Alessandro", "Palo", "Iolanda", "Helena"}


# the seed for random stuff
seed: int = 17
