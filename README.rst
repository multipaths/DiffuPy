
.. image:: https://github.com/multipaths/diffupy/blob/master/docs/source/meta/diffupy_logo.png
   :align: center
   :target: https://diffupy.readthedocs.io/en/latest/

Introduction |build| |docs|
===========================

DiffuPy is a generalizable Python implementation of the numerous label propagation algorithms. DiffuPy supports generic
graph formats such as `JSON, CSV, GraphML, or GML <https://github.com/multipaths/DiffuPy/tree/master/examples>`_. Check
out `DiffuPy's documentation here <https://diffupy.readthedocs.io/en/latest/>`_.

Installation
------------
The latest stable code can be installed from `PyPI <https://pypi.python.org/pypi/diffupy>`_ with:

.. code-block:: sh

   $ python3 -m pip install diffupy

The most recent code can be installed from the source on `GitHub <https://github.com/multipaths/DiffuPy>`_ with:

.. code-block:: sh

   $ python3 -m pip install git+https://github.com/multipaths/DiffuPy.git

For developers, the repository can be cloned from `GitHub <https://github.com/multipaths/DiffuPy>`_ and installed in
editable mode with:

.. code-block:: sh

   $ git clone https://github.com/multipaths/DiffuPy.git
   $ cd diffupy
   $ python3 -m pip install -e .

Command Line Interface
----------------------
The following commands can be used directly from your terminal:

1. **Run a diffusion analysis**
The following command will run a diffusion method on a given network with the given data.  More information `here
<https://diffupy.readthedocs.io/en/latest/diffusion.html>`_.

.. code-block:: sh

    $ python3 -m diffupy diffuse --network=<path-to-network-file> --data=<path-to-data-file> --method=<method>


2. **Generate a kernel with one of the seven methods implemented**
Generates the regularised Laplacian kernel of a given graph. More information in the `documentation
<https://diffupy.readthedocs.io/en/latest/kernels.html>`_.

.. code-block:: sh

    $ python3 -m diffupy kernel --network=<path-to-network-file>

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
