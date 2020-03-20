DiffuPy |release| Documentation
===============================
DiffuPy is a generalizable Python implementation of the numerous label propagation algorithms inspired by the diffuStats
R package [1]_. DiffuPy supports generic graph formats such as JSON, CSV, GraphML, or GML.

Installation is as easy as getting the code from `PyPI <https://pypi.python.org/pypi/diffupy>`_ with
:code:`python3 -m pip install diffupy`. See the :doc:`installation <installation>` documentation.

.. seealso::

    - Documented on `Read the Docs <http://diffupy.readthedocs.io/>`_
    - Versioned on `GitHub <https://github.com/multipaths/diffupy>`_
    - Tested on `Travis CI <https://travis-ci.org/multipaths/diffupy>`_
    - Distributed by `PyPI <https://pypi.python.org/pypi/diffupy>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   intro
   cli
   kernels
   diffusion
   constants
   matrix

References
----------
.. [1] Picart-Armada, S., *et al.* (2017). `Null diffusion-based enrichment for metabolomics data
   <https://doi.org/10.1371/journal.pone.0189012>`_. *PloS one* 12.12.

Disclaimer
----------
DiffuPy is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or
guarantee of maintenance, support, or back-up of data.

