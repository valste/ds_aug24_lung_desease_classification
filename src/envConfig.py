import sys, os
from pathlib import Path
from pprint import pprint

#---setting up the PYTHONPATH------

# Current directory as Path object
current_dir = Path.cwd()
# get project root folder
project_dir = str(current_dir.parent)
# adding project directory to the PYTHONPATH
sys.path.append(project_dir)
pprint(sys.path)


#----setting data directories----

from src import defs
defs.initDataPaths(project_dir)
defs.checkDataPaths()