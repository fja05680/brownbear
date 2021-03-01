"""
utils
---------
some useful utility functions
"""

# imports
import pandas as pd
from pathlib import Path
import os
from pandas_datareader.data import get_quote_yahoo
import brownbear as bb

# brownbear project root dir
ROOT = str(Path(os.getcwd().split('brownbear')[0] + '/brownbear'))

# symbol cache location
SYMBOL_CACHE = str(Path(ROOT + '/symbol-cache'))

class dotdict(dict):
    """ dot.notation access to dictionary attributes 
        mydict = {'val':'it works'}
        nested_dict = {'val':'nested works too'}
        mydict = dotdict(mydict)
        mydict.val
        # 'it works'

        mydict.nested = dotdict(nested_dict)
        mydict.nested.val
        #'nested works too'
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def get_quote(symbols):
    """ returns the current quote for a list of symbols as a dict """
    d = get_quote_yahoo(symbols).price.to_dict()
    return d

