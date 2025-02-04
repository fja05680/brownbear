"""
Utility functions.
"""

from pathlib import Path

import pandas as pd


########################################################################
# CONSTANTS

def _get_project_top_level():
    """
    Returns the outermost brownbear path.
    """
    current_path = Path(__file__).resolve()
    brownbear_path = None
    for parent in current_path.parents:
        if parent.name == "brownbear":
            brownbear_path = parent
    if brownbear_path:
        return brownbear_path
    raise RuntimeError("Top-level brownbear directory not found")

ROOT = _get_project_top_level()
"""
str : Full path to brownbear project root dir.
"""
SYMBOL_CACHE = ROOT / 'symbol-cache'
"""
str : Full path to symbol-cache dir.
"""


########################################################################
# FUNCTIONS

def csv_to_df(filepaths):
    """
    Read multiple csv files into a dataframe.

    Parameters
    ----------
    filepaths : list of str
        List of of full path to csv files.

    Returns
    -------
    df : pd.DataFrame
        Dataframe representing the concatination of the csv files
        listed in `filepaths`.
    """
    l = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, skip_blank_lines=True, comment='#')
        l.append(df)
    df = pd.concat(l)
    return df


class dotdict(dict):
    """
    Provides dot.notation access to dictionary attributes.

    Examples
    --------
    >>> mydict = {'val' : 'it works'}
    >>> mydict = dotdict(mydict)
    >>> mydict.val
    'it works'
    >>> nested_dict = {'val' : 'nested works too'}
    >>> mydict.nested = dotdict(nested_dict)
    >>> mydict.nested.val
    'nested works too'
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_full(x):
    """
    Print every row of list-like object.
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
