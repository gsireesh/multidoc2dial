#!/bin/sh

seg=$1 # token or structure
domain=$2 # dmv va ssa or studentaid
YOUR_DIR=../data # change it to your own local dir

python data_preprocessor.py \
--dataset_config_name multidoc2dial \
--output_dir $YOUR_DIR/mdd_dpr \
--segmentation $seg \
--dpr