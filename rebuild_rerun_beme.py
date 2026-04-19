import subprocess
import sys


subprocess.run([sys.executable, "rerun_beme_standard_ap.py"], check=True)
subprocess.run([sys.executable, "-m", "part_3_metrics_collection.tc_batch_runner"], check=True)
