# -*- coding: utf-8 -*-

"""
The goal of this package is to re-implement the Diffusion algorithm described in ... in Python.
TODO: Add me

Installation
------------
1. ``diffupy`` can be installed with the following commmands:

.. code-block:: sh

    python3 -m pip install ...

2. or in editable mode with:

.. code-block:: sh

    git clone ..

.. code-block:: sh

    cd diffupy

.. code-block:: sh

    python3 -m pip install -e .

How to use
----------

"""

import logging

log = logging.getLogger(__name__)

__version__ = '0.0.1-dev'

__title__ = 'diffupy'
__description__ = "The diffuStats package consists of functions to compute graph kernels, the function diffuse to " \
                  "compute the diffusion scores and the function perf_eval and its wrapper perf to compute performance " \
                  "measures."
__url__ = 'https://github.com/jmarinllao/diffupy'

__author__ = 'Josep Marín-Llaó, Sergi Picart Armada, Daniel Domingo-Fernández'
__email__ = 'josepmarinllao@gmail.com'

__license__ = 'Apache License'
__copyright__ = 'Copyright (c) 2019 Josep Marín-Llaó, Sergi Picart Armada, Daniel Domingo-Fernández'
