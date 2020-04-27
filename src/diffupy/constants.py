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
#: xml
XLS = 'xls'
#: xmls
XLSX = 'xlsx'
#: tsv
TSV = 'tsv'
#: graphML
GRAPHML = 'graphml'
#: bel
BEL = 'bel'
#: node link json
JSON = 'json'
#: pickle
PICKLE = 'pickle'
#: gml
GML = 'gml'
#: edge list
EDGE_LIST = '.lst'

XLS_FORMATS = (
    XLS,
    XLSX
)

#: Available graph formats
GRAPH_FORMATS = (
    CSV,
    TSV,
    GRAPHML,
    BEL,
    JSON,
    PICKLE,
)

#: Available kernel formats
KERNEL_FORMATS = (
    CSV,
    TSV,
    JSON,
    PICKLE,
)

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
LABEL = 'Label'
ENTITY = 'Entity'
GENE = 'Gene'

NODE_LABELING = [
    NODE,
    LABEL,
    ENTITY,
    GENE
]

#: Node type
NODE_TYPE = 'NodeType'
#: Unspecified score type
SCORE = 'Score'
#: Log2 fold change (logFC)
LOG_FC = 'LogFC'
#: Statistical significance (p-value)
P_VALUE = 'p-value'
