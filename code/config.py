from typing import *
from pathlib import Path
import sys

# define the path of the root folder indipendently from where the code is executed
ROOT_PATH: Path = Path(__file__).parent.parent

# append the folder to the path
sys.path.append(str(ROOT_PATH))
