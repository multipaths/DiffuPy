Here you can find samples of input datasets and networks to run diffusion methods over.

Input Data
----------

You can submit your dataset in any of the following formats:

- CSV (.csv)
- TSV (.tsv)

Please ensure the dataset has a column for each of the following:

- Node
- Expresssion [*]_
- p-value

Input dataset example
~~~~~~~~~~~~~~~~~~~~~

+--------------+------------+---------+
| Node         | Expression | p-value |
+==============+============+=========+
| Gene A       | 4          | 0.03    |
+--------------+------------+---------+
| Gene  B      | -1         | 0.05    |
+--------------+------------+---------+
| Metabolite C | 1.5        | 0.001   |
+--------------+------------+---------+
| Gene D       | 3          |  0.07   |
+--------------+------------+---------+

See the `sample datasets <https://github.com/multipaths/DiffuPy/tree/master/examples/datasets>`_ directory for example
files.

.. [*] Differential expression values e.g. fold change (FC)

Networks
--------

If you would like to submit your own networks, please ensure they are in one of the following formats:

- BEL_ (.bel)

- CSV (.csv)

- Edge_ `list`__ (.lst)

- GML_ (.gml or .xml)

- GraphML_ (.graphml or .xml)

- Pickle (.pickle)

- TSV (.tsv)

- TXT (.txt)

.. _Edge: https://networkx.github.io/documentation/stable/reference/readwrite/edgelist.html
__ Edge_
.. _GraphML: http://graphml.graphdrawing.org
.. _BEL: https://language.bel.bio/
.. _GML: http://docs.yworks.com/yfiles/doc/developers-guide/gml.html


Minimally, please ensure each of the following columns are included in the network file you submit:

- FirstNode
- SecondNode

Optionally, you can choose to add a third column, "Relation" in your network (as in the example below). If the relation
between the **Source** and **Target** nodes is omitted, and/or if the directionality is ambiguous, either node can be
assigned as the **Source** or **Target**.

Custom-network example
~~~~~~~~~~~~~~~~~~~~~~

+-----------+--------------+-------------+
| FirstNode | SecondNode   | Relation    |
+===========+==============+=============+
| Gene A    | Gene B       | Increase    |
+-----------+--------------+-------------+
| Gene B    | Metabolite C | Association |
+-----------+--------------+-------------+
| Gene A    | Pathology D  | Association |
+-----------+--------------+-------------+

See the `sample networks <https://github.com/multipaths/DiffuPy/tree/master/examples/networks>`_ directory for some
examples.

