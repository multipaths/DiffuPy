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

"""Available diffusion methods"""

#: raw
RAW = 'raw'
#: ml
ML = 'ml'
#: gm
GM = 'gm'
#: mc
MC = 'mc'
#: z
Z = 'z'
#: ber_s
BER_S = 'ber_s'
#: ber p
BER_P = 'ber_p'

#: DiffuPy diffusion methods
METHODS = {
    RAW,
    ML,
    GM,
    MC,
    Z,
    BER_S,
    BER_P,
}

"""Available formats"""

#: csv
CSV = 'csv'
#: tsv
TSV = 'tsv'
#: graphML
GRAPHML = 'graphml'
#: bel
BEL = 'bel'
#: node link json
NODE_LINK_JSON = 'json'
#: pickle
BEL_PICKLE = 'pickle'
#: gml
GML = 'gml'
#: edge list
EDGE_LIST = '.lst'

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

"""Optional parameters"""
#: Expression value threshold
THRESHOLD = 'threshold'
#: Absolute value of expression level
ABSOLUTE_VALUE_EXP = 'absolute_value'

"""Acceptable column names of user submitted network"""

#: Column name for source node
SOURCE = 'Source'
#: Column name for target node
TARGET = 'Target'
#: Column name for relation
RELATION = 'Relation '

"""Dataset column names"""

#: Node name
NODE = 'Node'
#: Node type
NODE_TYPE = 'NodeType'
#: Log2 fold change (logFC)
LOG_FC = 'LogFC'
#: Statistical significance (p-value)
P_VALUE = 'p-value'
#: Label
LABEL = 'Label'
