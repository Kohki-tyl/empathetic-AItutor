"""Run the shared SFT trainer with this folder's fixed configuration."""
from pathlib import Path
import runpy
import sys

folder = Path(__file__).resolve().parent
if '--config' in sys.argv[1:]:
    raise SystemExit('This entry point uses its own config.json')
sys.argv[1:1] = ['--config', str(folder / 'config.json')]
runpy.run_path(str(folder.parents[3] / 'pipelines/sft/v4/train_v4_sft.py'), run_name='__main__')
