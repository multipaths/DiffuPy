DiffuPy |build| |docs| |coverage|
=================================

DiffuPy is a generalizable Python implementation of the numerous label propagation algorithms inspired by the `diffuStats <https://github.com/b2slab/diffuStats>`_ R package [1]_.

Installation
------------
1. ``diffupy`` can be installed with the following commands:

.. code-block:: sh

    $ python3 -m pip install git+https://github.com/jmarinllao/diffuPy.git@master

2. or in editable mode with:

.. code-block:: sh

    $ git clone https://github.com/jmarinllao/diffuPy.git
    $ cd diffupy
    $ python3 -m pip install -e .

How to Use
----------

1. **Generate Kernel**

Generates the kernel of a given graph.

.. code-block:: sh

    $ python3 -m diffupy kernel


References
----------

.. [1] Picart-Armada, S., *et al.* (2017). `Null diffusion-based enrichment for metabolomics data <https://doi.org/10.1371/journal.pone.0189012>`_. *PloS one* 12.12.

.. |build| image:: https://travis-ci.com/multipaths/diffupy.svg?branch=master
    :target: https://travis-ci.com/multipaths/diffupy
    :alt: Build Status
    
.. |docs| image:: http://readthedocs.org/projects/diffupy/badge/?version=latest
    :target: https://diffupy.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |coverage| image:: https://codecov.io/gh/multipaths/diffupy/coverage.svg?branch=master
    :target: https://codecov.io/gh/multipaths/diffupy?branch=master
    :alt: Coverage Status
