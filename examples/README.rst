Here you can find samples of input datasets and networks to run diffusion methods over.

Input Data
----------

You can submit your dataset in any of the following formats:

- CSV (.csv)
- TSV (.tsv)

Please ensure that the dataset has a column 'Node' containing node IDs. If you only provide the node IDs, you must
also ensure your dataset has a column 'NodeType' indicating the entity type for each node. You can also optionally add
the following columns to your dataset:

- log :sub:`2`  fold change (LogFC)
- p-value

Input dataset examples
~~~~~~~~~~~~~~~~~~~~~~

DiffuPath accepts several input formats which can be codified in different ways. See the
`diffusion scores <https://github.com/multipaths/DiffuPy/blob/master/docs/source/diffusion.rst>`_ summary for more
details.

1. You can provide a dataset with a column 'Node' containing node IDs along with a column 'NodeType' indicating the entity type.

+--------------+------------+
|   NodeType   |    Node    |
+==============+============+
|     Gene     |     A      |
+--------------+------------+
|     Gene     |     B      |
+--------------+------------+
|  Metabolite  |     C      |
+--------------+------------+
|     Gene     |     D      |
+--------------+------------+

2. You can also choose to provide a dataset with a column 'Node' containing node IDs as well as a column 'logFC' with
their log :sub:`2` FC.

+--------------+------------+
| Node         |   LogFC    |
+==============+============+
| Gene A       | 4          |
+--------------+------------+
| Gene  B      | -1         |
+--------------+------------+
| Metabolite C | 1.5        |
+--------------+------------+
| Gene D       | 3          |
+--------------+------------+

3. Finally, you can provide a dataset with a column 'Node' containing node IDs, a column 'logFC' with their log :sub:`2`
FC and a column 'p-value' with adjusted p-values.

+--------------+------------+---------+
| Node         |   LogFC    | p-value |
+==============+============+=========+
| Gene A       | 4          | 0.03    |
+--------------+------------+---------+
| Gene  B      | -1         | 0.05    |
+--------------+------------+---------+
| Metabolite C | 1.5        | 0.001   |
+--------------+------------+---------+
| Gene D       | 3          | 0.07    |
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
.. _BEL: https://language.bel.bio/
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
|  Source   |   Target     | Relation    |
+===========+==============+=============+
| Gene A    | Gene B       | Increase    |
+-----------+--------------+-------------+
| Gene B    | Metabolite C | Association |
+-----------+--------------+-------------+
| Gene A    | Pathology D  | Association |
+-----------+--------------+-------------+

See the `sample networks <https://github.com/multipaths/DiffuPy/tree/master/examples/networks>`_ directory for some
examples.
