"""
module: io.py

This module contains file handling functions for reading and writing files. It
enhances modularity and convenience in managing file operations.
"""

import pathlib
from typing import Any

import pandas as pd


def _convert_to_pathlib_object(str_path: str) -> pathlib.Path():
    """Converts string path into a pathlib.Path object

    Parameters
    ----------
    str_path : str
        string path that will be converted to a pathlib.Path object

    Returns
    -------
    pathlib.Path
        path to file or directory

    Raises
    ------
    FileNotFoundError
        Raised if the given `str_path` does not exist
    """
    return pathlib.Path(str_path).resolve(strict=True)


def _is_valid_path(value: Any) -> None | pathlib.Path:
    """Checks if the provided objects is a valid path.

    The function transforms strings into pathlib.Path objects to
    ensure consistent path formatting and maintain uniformity in
    path handling.

    Parameters
    ----------
    value : Any
        _description_

    Returns
    -------
    pathlib.Path
        None or converted string path into pathlib.Path object

    Raise
    -----
    TypeError
        raised if the `object` is not a string or pathlib.Path object
    FileNotFoundError
        raised if the `object` contains a path that does not exist
    """

    # checking if object is a str or pathlib.Path object
    if not isinstance(value, (str, pathlib.Path)):
        raise TypeError(
            "`object` must be a string or pathlib.Path object"
            f"Provided: {type(value)}"
        )
    if isinstance(value, str):
        value = _convert_to_pathlib_object(value)
        return value
    if isinstance(value, pathlib.Path):
        value = value.resolve(strict=True)
        return value


def load_data(fpath: str | pathlib.Path) -> pd.DataFrame:
    """Loads in image-base profiles as a pandas dataframe

    Parameters
    ----------
    fpath : str | pathlib.Path
        path to dataset

    Returns
    -------
    pd.DataFrame
        loaded image-based profile
    """

    # type checking
    fpath = _is_valid_path(fpath)

    # loading into pandas dataframe
    file_ext = fpath.suffix
    if file_ext == ".csv":
        df = pd.read_csv(fpath)
    elif file_ext == ".parquet":
        df = pd.read_parquet(fpath)
    else:
        raise TypeError(f"Unsupported file ext: {file_ext}")

    return df
