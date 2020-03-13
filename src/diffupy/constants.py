# -*- coding: utf-8 -*-

"""Constants of diffupy."""

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)))

#: Default DiffuPy directory
DEFAULT_DIFFUPY_DIR = os.path.join(os.path.expanduser('~'), '.diffupy')
#: Default DiffuPy output directory
OUTPUT = os.path.join(DEFAULT_DIFFUPY_DIR, 'output')


def ensure_output_dirs():
    """Ensure that the output directories exists."""
    os.makedirs(DEFAULT_DIFFUPY_DIR, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)


ensure_output_dirs()

#  Available methods for diffusion, as a character vector
#  Check diffuse docs for the detailed explanation of each

#: DiffuPy emoji
EMOJI = "🌐"

#: DiffuPy diffusion methods
METHODS = {
    "raw",
    "ml",
    "gm",
    "mc",
    "z",
    "ber_s",
    "ber_p",
}


"""Available formats"""

CSV = 'csv'
TSV = 'tsv'
GRAPHML = 'graphml'
BEL = 'bel'
NODE_LINK_JSON = 'json'
BEL_PICKLE = 'pickle'
GML = 'gml'

#: DiffuPath available network formats
FORMATS = [
    CSV,
    TSV,
    GRAPHML,
    BEL,
    NODE_LINK_JSON,
    BEL_PICKLE,
]

#: Separators
FORMAT_SEPARATOR_MAPPING = {
    CSV: ',',
    TSV: '\t'
}

"""Acceptable column names of user submitted network"""

#: Column name for source node
SOURCE = 'source'
#: Column name for target node
TARGET = 'target'
#: Column name for relation
RELATION = 'relation '
