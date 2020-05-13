#!/usr/bin/env bash

# network=network
# data=data file
# output=output file results
# method=diffusion method (i.e., gm, ml, raw and z)
# binarize=binarize labels (default True)
# absolute_value=apply absolute value to logFC in data (default True)
# threshold=threshold to apply if logFC in data
# p_value=statistical significance (default 0.05)

# TODO: Set here your variable pointing to the repo
pkg_path=/Users/danieldomingo/PycharmProjects/DiffuPy

echo "Running example 2 of DiffuPy "

python3 -m diffupy diffuse \
  --network ${pkg_path}/examples/networks/sample_network_2.csv \
  --input ${pkg_path}/examples/datasets/sample_dataset_with_logfc.csv \
  --method raw \
  --binarize True \
  --absolute_value True \
  --threshold 0.5 \
  --p_value 0.05

