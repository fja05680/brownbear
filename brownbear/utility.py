"""
Utility functions.
"""

import io
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

WIKIPEDIA_USER_AGENT = (
    'brownbear/1.0 (portfolio research; https://github.com/brownbear)'
)


########################################################################
# CONSTANTS

def _get_project_top_level():
    """
    Return the outermost brownbear project root directory.

    Returns
    -------
    Path
        Repository root directory named ``brownbear``.

    Raises
    ------
    RuntimeError
        If no parent directory named ``brownbear`` is found.
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

def _wikipedia_slug(title):
    """
    Build a Wikipedia article path from a page title.

    Parameters
    ----------
    title : str
        Wikipedia page title.

    Returns
    -------
    str
        URL-encoded article slug.
    """
    return quote(title.replace(' ', '_'), safe='_')


def get_wikipedia_table(title, filename, match, use_cache=False):
    """
    Fetch a table from a Wikipedia page and cache it as CSV.

    Wikipedia blocks API clients without a descriptive User-Agent. This
    fetches the rendered HTML page directly instead of using the
    ``wikipedia`` package.

    Parameters
    ----------
    title : str
        Wikipedia page title, e.g. ``'Dow Jones Industrial Average'``.
    filename : str or Path
        CSV file used to cache the table.
    match : str
        Substring passed to :func:`pandas.read_html` to select the table.
    use_cache : bool, optional
        When True and ``filename`` exists, skip the Wikipedia fetch.

    Returns
    -------
    pd.DataFrame
    """
    filename = Path(filename)
    if not (use_cache and filename.is_file()):
        slug = _wikipedia_slug(title)
        url = f'https://en.wikipedia.org/wiki/{slug}'
        response = requests.get(
            url,
            headers={'User-Agent': WIKIPEDIA_USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        df = pd.read_html(io.StringIO(response.text), header=0, match=match)[0]
        df.to_csv(filename, header=True, index=False, encoding='utf-8')

    return pd.read_csv(filename)


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

    Parameters
    ----------
    x
        Pandas object or other value passed to ``print``.

    Returns
    -------
    None
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def notebook_display_options(float_format='{:0.2f}'):
    """
    Configure pandas display for notebooks with large dataframes.

    Pair with the standard Jupyter javascript cell that disables output
    scrolling when reviewing long tables.

    Parameters
    ----------
    float_format : str, optional
        Format string for floating-point columns (default is ``'{:0.2f}'``).

    Returns
    -------
    None
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = float_format.format
