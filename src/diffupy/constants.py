"""Constants of diffuPy."""

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(SOURCE_DIR, os.pardir)), 'data')
KERNEL_DIR = os.path.join(DATA_DIR, 'kernels')


def ensure_output_dirs():
    """Ensure that the output directories exists."""
    os.makedirs(KERNEL_DIR, exist_ok=True)


# ' Available methods for diffusion
# '
# ' .available_methods is a character vector with the implemented scores
# '
# ' @rdname checks

# TODO: Change to a dictionary where keys are better explanatory terms of each method
METHODS = {
    "raw",
    "ml",
    "gm",
    "mc",
    "z",
    "ber_s",
    "ber_p"
}
