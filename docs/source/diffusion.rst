Diffusion
=========
The methods in this modules manage the treatment of the different score diffusion methods applied of/from an path set of
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
  nodes introduce null diffusion {y_raw[j] = 0}. [Vandin, 2011]. They are computed as: f_{raw} = K · y_{raw}. Where K is
  a graph kernel, see :doc:`kernel <kernel>`. These scores treat negative and unlabelled nodes equivalently.

- **ml**: Same as raw, but negative nodes introduce a negative unit of flow. Therefore not equivalent to unlabelled
  nodes. [Zoidi, 2015]

- **gl**: Same as ml, but the unlabelled nodes are assigned a (generally non-null) bias term based on the total number
  of positives, negatives and unlabelled nodes [Mostafavi, 2008].

- **ber_s**: A quantification of the relative change in the node score before and after the network smoothing. The score
  for a particular node i can be written as f_{ber_s}[i] = f_{raw}[i] / (y_{raw}[i] + eps). Where eps is a parameter
  controlling the importance of the relative change.

Methods with statistical normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **z**: a parametric alternative to the raw score of node is subtracted its mean value and divided by its standard
  deviation. Differential trait of this package. The statistical moments have a closed analytical form and are inspired
  in [Harchaoui, 2013].

- **mc**:  the score of node code {i} is based on its empirical p-value, computed by permuting the path {n.perm} times.
  It is roughly the proportion of path permutations that led to a diffusion score as high or higher than the original
  diffusion score.

- **ber_p**: A used in [Bersanelli, 2016], this score combines raw and mc, in order to take into account both the
  magnitude of the {raw} scores and the effect of the network topology: this is a quantification of the relative change
  in the node score before and after the network smoothing.


Summary table of methods without statistical normalization
----------------------------------------------------------
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+
| Scores      | y+       | y-       | yn       | Normalized    | Stochastic     | Quantitative   | Reference           |
+=============+==========+==========+==========+===============+================+================+=====================+
| raw         | 1        | 0        | 0        | No            | No.            | Yes            | Vandin (2010)       |
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+
| ml          | 1        | -1       | 0        | No            | No             | No             | Tsuda (2010)        |
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+
| gm          | 1        | -1       | k        | No            | No             | No             | Mostafavi (2008)    |
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+
| ber_s       | 1        | 0        | 0        | No            | No             | Yes            | Bersanelli (2016)   |
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+

Summary table of methods with statistical normalization
-------------------------------------------------------
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+
| Scores      | y+       | y-       | yn       | Normalized    | Stochastic     | Quantitative   | Reference           |
+=============+==========+==========+==========+===============+================+================+=====================+
| ber_p       | 1        | 0        | 0*       | Yes           | Yes            | Yes            | Bersanelli (2016)   |
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+
| mc          | 1        | 0        | 0*       | Yes           | Yes            | Yes            | Bersanelli (2016)   |
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+
| z           | 1        | 0        | 0*       | Yes           | No             | Yes            | Harchaoui (2013)    |
+-------------+----------+----------+----------+---------------+----------------+----------------+---------------------+


.. automodule:: diffupy.diffusion
   :members:

.. automodule:: diffupy.diffusion_raw
   :members: