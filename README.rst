DiffuPy |build| |docs| |coverage|
=================================

DiffuPy is a generalizable Python implementation of the numerous label propagation algorithms inspired by the `diffuStats <https://github.com/b2slab/diffuStats>`_ R package [1]_.

Installation
------------
1. ``diffupy`` can be installed with the following commands:

.. code-block:: sh

    $ python3 -m pip install git+https://github.com/multipaths/DiffuPy.git@master

2. or in editable mode with:

.. code-block:: sh

    $ git clone https://github.com/multipaths/DiffuPy.git
    $ cd diffupy
    $ python3 -m pip install -e .

Command Line Interface
----------------------
The following commands can be used directly use from your terminal:

1. **Run a diffusion analysis**
The following command will run a diffusion method on a given network with the given data

.. code-block:: sh

    $ python3 -m diffupy diffuse --network=<path-to-network-file> --input=<path-to-data-file> --method=<method>


2. **Generate a kernel with one of the seven methods implemented**
Generates the regularised Laplian kernel of a given graph.

.. code-block:: sh

    $ python3 -m diffupy kernel --network=<path-to-network-file>


References
----------
.. [1] Picart-Armada, S., *et al.* (2017). `Null diffusion-based enrichment for metabolomics data
<https://doi.org/10.1371/journal.pone.0189012>`_. *PloS one* 12.12.

Disclaimer
----------
DiffuPy is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or
guarantee of maintenance, support, or back-up of data.

.. |build| image:: https://travis-ci.com/multipaths/diffupy.svg?branch=master
    :target: https://travis-ci.com/multipaths/diffupy
    :alt: Build Status

.. |docs| image:: http://readthedocs.org/projects/diffupy/badge/?version=latest
    :target: https://diffupy.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |coverage| image:: https://codecov.io/gh/multipaths/diffupy/coverage.svg?branch=master
    :target: https://codecov.io/gh/multipaths/diffupy?branch=master
    :alt: Coverage Status
