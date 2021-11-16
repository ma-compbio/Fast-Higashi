#!/bin/bash

python fast_higashi2_ztm_concat.py -c ../config/config_ramani.JSON --path2input_cache ramani_1m \
  --path2result_dir ramani_1m_para_final --use_intra --method PARAFAC2 --extra conv_rwr --do_conv --do_rwr
