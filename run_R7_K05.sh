#!/bin/bash
# 4-view R7 (kappa=0.05/pair) — 12-slice DLPFC
# PROVEN CONFIG: ARI=0.5763 on 151507
cd /home/bio/lhz/spatialGDC
find spCLUE -name '*.pyc' -delete 2>/dev/null
rm -rf spCLUE/__pycache__
/home/bio/miniconda3/envs/spCLUE/bin/python -u run_4view_DLPFC.py
