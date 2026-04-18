import subprocess
import sys

subprocess.run([sys.executable, "rebuild_standard_beme_trees.py"], check=True)
subprocess.run([sys.executable, "rerun_beme_standard.py"], check=True)