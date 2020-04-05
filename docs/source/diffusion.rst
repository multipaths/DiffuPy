Diffusion
=========
The methods in this modules manage the treatment of the different score diffusion methods applied to/from a path set of
labels/scores of/on a certain network (as a graph format or a graph kernel matrix stemming from a graph).

Diffusion methods procedures provided in this package differ on:
(a) How to distinguish positives, negatives and unlabelled examples.
(b) Their statistical normalisation.

Input scores can be specified in three formats:
1. A named numeric vector, whereas if several of these vectors that share the node names need to be smoothed.
2. A column-wise matrix. However, if the unlabelled entities are not the same from one case to another.
2. A named list of such score matrices can be passed to this function. The path format will be kept in the output.

If the path labels are not quantitative, i.e. positive(1), negative(0) and possibly unlabelled, all the scores raw, gm,
ml, z, mc, ber_s, ber_p can be used.

Methods
-------

Methods without statistical normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **raw**: positive nodes introduce unitary flow {y_raw[i] = 1} to the network, whereas either negative and unlabelled
  nodes introduce null diffusion {y_raw[j] = 0}. [1]_. They are computed as: f_{raw} = K · y_{raw}. Where K is
  a graph kernel, see :doc:`kernels <kernels>`. These scores treat negative and unlabelled nodes equivalently.

- **ml**: Same as raw, but negative nodes introduce a negative unit of flow. Therefore not equivalent to unlabelled
  nodes. [2]_

- **gl**: Same as ml, but the unlabelled nodes are assigned a (generally non-null) bias term based on the total number
  of positives, negatives and unlabelled nodes [3]_.

- **ber_s**: A quantification of the relative change in the node score before and after the network smoothing. The score
  for a particular node i can be written as f_{ber_s}[i] = f_{raw}[i] / (y_{raw}[i] + eps). Where eps is a parameter
  controlling the importance of the relative change.

Methods with statistical normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **z**: a parametric alternative to the raw score of node is subtracted its mean value and divided by its standard
  deviation. Differential trait of this package. The statistical moments have a closed analytical form and are inspired
  in [4]_.

- **mc**: the score of node code {i} is based on its empirical p-value, computed by permuting the path {n.perm} times.
  It is roughly the proportion of path permutations that led to a diffusion score as high or higher than the original
  diffusion score.

- **ber_p**: used in [5]_, this score combines raw and mc, in order to take into account both the
  magnitude of the {raw} scores and the effect of the network topology: this is a quantification of the relative change
  in the node score before and after the network smoothing.


Summary tables
--------------
Methods without statistical normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+
| Scores      | y+       | y-      | yn     | Normalized  | Stochastic  | Quantitative    | Reference  |
+=============+==========+=========+========+=============+=============+=================+============+
| raw         | 1        | 0       | 0      | No          | No.         | Yes             |    [1]_    |
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+
| ml          | 1        | -1      | 0      | No          | No          | No              |    [6]_    |
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+
| gm          | 1        | -1      | k      | No          | No          | No              |    [3]_    |
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+
| ber_s       | 1        | 0       | 0      | No          | No          | Yes             |    [5]_    |
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+

Methods with statistical normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+
| Scores      | y+       | y-      | yn     | Normalized  | Stochastic  | Quantitative    | Reference  |
+=============+==========+=========+========+=============+=============+=================+============+
| ber_p       | 1        | 0       | 0*     | Yes         | Yes         | Yes             |    [5]_    |
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+
| mc          | 1        | 0       | 0*     | Yes         | Yes         | Yes             |    [5]_    |
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+
| z           | 1        | 0       | 0*     | Yes         | No          | Yes             |    [4]_    |
+-------------+----------+---------+--------+-------------+-------------+-----------------+------------+


.. automodule:: diffupy.diffuse
   :members:

.. automodule:: diffupy.diffuse_raw
   :members:

References
----------
.. [1] Vandin, F., *et al.* (2010). Algorithms for detecting significantly mutated pathways in cancer. Lecture Notes in
    Computer Science. 6044, 506–521.

.. [2] Zoidi, O., *et al.* (2015). Graph-based label propagation in digital media: A review. ACM Computing Surveys
    (CSUR), 47(3), 1-35.

.. [3] Mostafavi, S., *et al.* (2008). Genemania: a real-time multiple association network integration algorithm for
    predicting gene function.Genome Biology. (9), S4.

.. [4] Harchaoui, Z., *et al.* (2013). Kernel-based methods for hypothesis testing: a unified view. IEEE Signal
    Processing Magazine. (30), 87–97.

.. [5] Bersanelli, M. *et al.* (2016). Network diffusion-based analysis of high-throughput data for the detection of
    differentially enriched modules. Scientific Reports. (6), 34841.

.. [6] Tsuda, K., *et al.* (2005).  Fast  protein  classification  with  multiple  networks. Bioinformatics, (21), 59–65


