#!/bin/bash
cd /home/claude/f1model
for m in base gap; do [ -f outputs_v2/gap/${m}_pairs.pkl ] || PYTHONPATH=. python3 scripts/link_test.py $m; done
