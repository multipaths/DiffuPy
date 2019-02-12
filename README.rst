diffuPy |build| |coverage|
==========================

diffuPy is a generalizable Python implementation of the null diffusion algorithm for metabolomics data described by [1].
The package enables running the algorithm in heterogenous networks by abstracting `NetworkX <http://networkx.github.io/>`_ classes.

Citation
--------
If you use diffuPy in your work, please cite the R implementation of the null diffusion algorithm [1]_ (more info in `diffuStats <https://github.com/b2slab/diffuStats>`_):

.. [1] Picart-Armada, S., *et al.* (2019). `Null diffusion-based enrichment for metabolomics data < https://doi.org/10.1371/journal.pone.0189012>`_. *PloS one* 12.12 (2017):.

.. |build| image:: https://travis-ci.com/jmarinllao/diffupy.svg?branch=master
    :target: https://travis-ci.com/jmarinllao/diffupy
    :alt: Build Status

.. |coverage| image:: https://codecov.io/gh/jmarinllao/diffupy/coverage.svg?branch=master
    :target: https://codecov.io/gh/jmarinllao/diffupy?branch=master
    :alt: Coverage Status
