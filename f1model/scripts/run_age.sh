#!/bin/bash
cd /home/claude/f1model
for m in base age; do [ -f outputs_v2/gap/${m}_long_pairs.pkl ] || LINK_W=800 LINK_T=8 LINK_SUF=_long PYTHONPATH=. python3 scripts/link_test.py $m; done
