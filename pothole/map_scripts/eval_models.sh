#!/bin/bash

TOTAL_SAMPLES=`ls data/processed/detection/ground-truth | wc -l`

echo "There are $TOTAL_SAMPLES images in test set" > /tmp/eval_vpu.txt

echo "SSD FPN OPENVINO VPU(no bifocal)" >> /tmp/eval_vpu.txt
python -m write_predictions --model ssd_fpn_openvino_cpu --bifocal 0 --bin_only 1
python -m compute_map --no-plot >> /tmp/eval_vpu.txt

