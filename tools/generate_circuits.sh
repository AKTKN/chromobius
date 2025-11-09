#!/bin/bash

set -e
set -o pipefail

parallel --ungroup ./tools/gen_circuits \
    --style {1} \
    --out_dir /home/quantum_teresheys/workspace/qldpc_bp_project/qldpc_bp/src/qldpc_bp/data/circuits/colorcode \
    --noise_model {2} \
    --diameter {3} \
    --rounds "d" \
    --noise_strength {4} \
    ::: superdense_color_code_Z \
    ::: uniform si1000 \
    ::: 3 5 7 9 11 13 15 17 19 21 \
    ::: 0.0005 0.0008 0.001 0.002 0.003
