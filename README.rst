
.. image:: https://github.com/multipaths/diffupy/blob/master/docs/source/meta/diffupy_logo.png
   :align: center
   :target: https://diffupy.readthedocs.io/en/latest/

Introduction |build| |docs| |zenodo|
====================================

DiffuPy is a generalizable Python implementation of the numerous label propagation algorithms. DiffuPy supports generic
graph formats such as `JSON, CSV, GraphML, or GML <https://github.com/multipaths/DiffuPy/tree/master/examples>`_. Check
out `DiffuPy's documentation here <https://diffupy.readthedocs.io/en/latest/>`_.

Installation |pypi_version| |python_versions| |pypi_license|
------------------------------------------------------------
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

Here you can find samples of input datasets and networks to run diffusion methods over.

Input Data
----------

You can submit your dataset in any of the following formats:

- CSV (.csv)
- TSV (.tsv)

Please ensure that the dataset minimally has a column 'Node' containing node IDs. You can also optionally add the
following columns to your dataset:

- NodeType
- LogFC [*]_
- p-value

.. [*] |Log| fold change

.. |Log| replace:: Log\ :sub:`2`

Input dataset examples
~~~~~~~~~~~~~~~~~~~~~~

DiffuPath accepts several input formats which can be codified in different ways. See the
`diffusion scores <https://github.com/multipaths/DiffuPy/blob/master/docs/source/diffusion.rst>`_ summary for more
details.

1. You can provide a dataset with a column 'Node' containing node IDs.

+------------+
|     Node   |
+============+
|      A     |
+------------+
|      B     |
+------------+
|      C     |
+------------+
|      D     |
+------------+

2. You can also provide a dataset with a column 'Node' containing node IDs as well as a column 'NodeType', indicating
the entity type of the node to run diffusion by entity type.

+------------+--------------+
|     Node   |   NodeType   |
+============+==============+
|      A     |     Gene     |
+------------+--------------+
|      B     |     Gene     |
+------------+--------------+
|      C     |  Metabolite  |
+------------+--------------+
|      D     |    Gene      |
+------------+--------------+

3. You can also choose to provide a dataset with a column 'Node' containing node IDs as well as a column 'logFC' with
their logFC. You may also add a 'NodeType' column to run diffusion by entity type.

+--------------+------------+
| Node         |   LogFC    |
+==============+============+
|      A       | 4          |
+--------------+------------+
|      B       | -1         |
+--------------+------------+
|      C       | 1.5        |
+--------------+------------+
|      D       | 3          |
+--------------+------------+

4. Finally, you can provide a dataset with a column 'Node' containing node IDs, a column 'logFC' with their logFC and a
column 'p-value' with adjusted p-values. You may also add a 'NodeType' column to run diffusion by entity type.

+--------------+------------+---------+
| Node         |   LogFC    | p-value |
+==============+============+=========+
|      A       | 4          | 0.03    |
+--------------+------------+---------+
|      B       | -1         | 0.05    |
+--------------+------------+---------+
|      C       | 1.5        | 0.001   |
+--------------+------------+---------+
|      D       | 3          | 0.07    |
+--------------+------------+---------+

See the `sample datasets <https://github.com/multipaths/DiffuPy/tree/master/examples/datasets>`_ directory for example
files.

Networks
--------

If you would like to submit your own networks, please ensure they are in one of the following formats:

- BEL_ (.bel)

- CSV (.csv)

- Edge_ `list`__ (.lst)

- GML_ (.gml or .xml)

- GraphML_ (.graphml or .xml)

- Pickle (.pickle). BELGraph object from PyBEL_ 0.13.2

- TSV (.tsv)

- TXT (.txt)

.. _Edge: https://networkx.github.io/documentation/stable/reference/readwrite/edgelist.html
__ Edge_
.. _GraphML: http://graphml.graphdrawing.org
.. _BEL: https://biological-expression-language.github.io
.. _GML: http://docs.yworks.com/yfiles/doc/developers-guide/gml.html
.. _PyBEL: https://github.com/pybel/pybel/


Minimally, please ensure each of the following columns are included in the network file you submit:

- Source
- Target

Optionally, you can choose to add a third column, "Relation" in your network (as in the example below). If the relation
between the **Source** and **Target** nodes is omitted, and/or if the directionality is ambiguous, either node can be
assigned as the **Source** or **Target**.

Custom-network example
~~~~~~~~~~~~~~~~~~~~~~

+-----------+--------------+-------------+
|  Source   |   Target     |  Relation   |
+===========+==============+=============+
|     A     |      B       | Increase    |
+-----------+--------------+-------------+
|     B     |      C       | Association |
+-----------+--------------+-------------+
|     A     |      D       | Association |
+-----------+--------------+-------------+

See the `sample networks <https://github.com/multipaths/DiffuPy/tree/master/examples/networks>`_ directory for some
examples.


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

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/diffupy.svg
    :alt: Stable Supported Python Versions

.. |pypi_version| image:: https://img.shields.io/pypi/v/diffupy.svg
    :alt: Current version on PyPI

.. |pypi_license| image:: https://img.shields.io/pypi/l/diffupy.svg
    :alt: Apache-2.0

..  |zenodo| image:: https://zenodo.org/badge/195810310.svg
   :target: https://zenodo.org/badge/latestdoi/195810310

