import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin
from clint.textui import progress
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from struct import unpack
from collections import namedtuple, Counter
from datetime import timedelta
from time import time


def maybe_download(url):
    """Download and unzip ITCH data if not yet available"""
    filename = data_path / url.split("/")[-1]
    if not data_path.exists():
        print("Creating directory")
        data_path.mkdir()
    if not filename.exists():
        print("Downloading...", url)
        urlretrieve(url, filename)
    unzipped = data_path / (filename.stem + ".bin")
    if not unzipped.exists():
        print("Unzipping to", unzipped)
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return unzipped


data_path = Path('data')
itch_store = str(data_path / "itch.h5")
order_book_store = data_path / "order_book.h5"

FTP_URL = "ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/"
SOURCE_FILE = "03272019.NASDAQ_ITCH50.gz"

filename = maybe_download(urljoin(FTP_URL, SOURCE_FILE))
date = filename.name.split('.')[0]

event_codes = {'O': 'Start of Messages',
               'S': 'Start of System Hours',
               'Q': 'Start of Market Hours',
               'M': 'End of Market Hours',
               'E': 'End of System Hours',
               'C': 'End of Messages'}

encoding = {'primary_market_maker': {'Y': 1, 'N': 0},
            'printable'           : {'Y': 1, 'N': 0},
            'buy_sell_indicator'  : {'B': 1, 'S': -1},
            'cross_type'          : {'O': 0, 'C': 1, 'H': 2},
            'imbalance_direction' : {'B': 0, 'S': 1, 'N': 0, 'O': -1}}

formats = {
    ('integer', 2): 'H',
    ('integer', 4): 'I',
    ('integer', 6): '6s',
    ('integer', 8): 'Q',
    ('alpha', 1)  : 's',
    ('alpha', 2)  : '2s',
    ('alpha', 4)  : '4s',
    ('alpha', 8)  : '8s',
    ('price_4', 4): 'I',
    ('price_8', 8): 'Q',
}

message_data = (pd.read_excel('message_types.xlsx', sheet_name='messages', encoding='latin1')
                .sort_values('id')
                .drop('id', axis=1))

print("yuck")
