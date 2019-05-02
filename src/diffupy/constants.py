"""Constants of diffuPy."""

# ' Available methods for diffusion
# '
# ' .available_methods is a character vector with the implemented scores
# '
# ' @rdname checks
# TODO: Change to a dictionary where keys are better explanatory terms of each method
METHODS = [
    "raw",
    "ml",
    "gm",
    "mc",
    "z",
    "ber_s",
    "ber_p"
]
