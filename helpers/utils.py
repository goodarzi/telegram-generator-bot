import sys
import time
import math
from datetime import datetime
from ruamel.yaml import YAML


def load_config(config_file):
    """Load user config"""
    yaml = YAML()
    with open(config_file, encoding="utf-8") as f:
        config = yaml.load(f)
        if config:
            return config
        else:
            return None


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def format_byte(size: float, dot=2):
    """format byte"""

    # pylint: disable = R0912
    if 0 <= size < 1:
        human_size = str(round(size / 0.125, dot)) + "b"
    elif 1 <= size < 1024:
        human_size = str(round(size, dot)) + "B"
    elif math.pow(1024, 1) <= size < math.pow(1024, 2):
        human_size = str(round(size / math.pow(1024, 1), dot)) + "KB"
    elif math.pow(1024, 2) <= size < math.pow(1024, 3):
        human_size = str(round(size / math.pow(1024, 2), dot)) + "MB"
    elif math.pow(1024, 3) <= size < math.pow(1024, 4):
        human_size = str(round(size / math.pow(1024, 3), dot)) + "GB"
    elif math.pow(1024, 4) <= size < math.pow(1024, 5):
        human_size = str(round(size / math.pow(1024, 4), dot)) + "TB"
    elif math.pow(1024, 5) <= size < math.pow(1024, 6):
        human_size = str(round(size / math.pow(1024, 5), dot)) + "PB"
    elif math.pow(1024, 6) <= size < math.pow(1024, 7):
        human_size = str(round(size / math.pow(1024, 6), dot)) + "EB"
    elif math.pow(1024, 7) <= size < math.pow(1024, 8):
        human_size = str(round(size / math.pow(1024, 7), dot)) + "ZB"
    elif math.pow(1024, 8) <= size < math.pow(1024, 9):
        human_size = str(round(size / math.pow(1024, 8), dot)) + "YB"
    elif math.pow(1024, 9) <= size < math.pow(1024, 10):
        human_size = str(round(size / math.pow(1024, 9), dot)) + "BB"
    elif math.pow(1024, 10) <= size < math.pow(1024, 11):
        human_size = str(round(size / math.pow(1024, 10), dot)) + "NB"
    elif math.pow(1024, 11) <= size < math.pow(1024, 12):
        human_size = str(round(size / math.pow(1024, 11), dot)) + "DB"
    elif math.pow(1024, 12) <= size:
        human_size = str(round(size / math.pow(1024, 12), dot)) + "CB"
    else:
        raise ValueError(
            f'format_byte() takes number than or equal to 0, " \
            " but less than 0 given. {size}'
        )
    return human_size


def create_progress_bar(progress, total_bars=10):
    """
    example
    progress = 50
    progress_bar = create_progress_bar(progress)
    print(f'Progress: [{progress_bar}] ({progress}%)')
    """
    completed_bars = int(progress * total_bars / 100)
    remaining_bars = total_bars - completed_bars
    progress_bar = "█" * completed_bars + "░" * remaining_bars
    return progress_bar
