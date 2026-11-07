#!/bin/bash
cd /home/claude/f1model
for m in off on; do [ -f outputs_v2/link/${m}_pairs.pkl ] || PYTHONPATH=. python3 scripts/link_test.py $m; done
