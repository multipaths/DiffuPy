"""Constants of diffupy."""

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)))

DEFAULT_DIFFUPY_DIR = os.path.join(os.path.expanduser('~'), '.diffupy')
OUTPUT = os.path.join(DEFAULT_DIFFUPY_DIR, 'output')


def ensure_output_dirs():
    """Ensure that the output directories exists."""
    os.makedirs(DEFAULT_DIFFUPY_DIR, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)


ensure_output_dirs()

# TODO: Change to a dictionary where keys are better explanatory terms of each methode.
#  ANSWER: The explanation is long and tedious, it is on diffuse.py header documentation.
#  Should I provide it here too?

#  Available methods for diffusion, as a character vector
#  Check diffuse docs for the detailed explanation of each

METHODS = {
    "raw",
    "ml",
    "gm",
    "mc",
    "z",
    "ber_s",
    "ber_p"
}
