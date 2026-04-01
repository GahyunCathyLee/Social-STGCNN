#!/usr/bin/env bash
set +e

python3 train.py --config configs/c0-1.yaml

python3 train.py --config configs/c0-2.yaml

python3 train.py --config configs/c0-3.yaml

python3 train.py --config configs/c0-4.yaml

python3 train.py --config configs/c0-5.yaml

python3 train.py --config configs/c2-1.yaml

python3 train.py --config configs/c2-2.yaml

python3 train.py --config configs/c2-3.yaml

python3 train.py --config configs/c2-4.yaml

python3 train.py --config configs/c2-5.yaml