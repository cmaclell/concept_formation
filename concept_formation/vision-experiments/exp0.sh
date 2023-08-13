#!/bin/bash

# 0
python3 main_mnist.py --experiment=0 --label=0 --model-type='cobweb' --seed=123
python3 main_mnist.py --experiment=0 --label=0 --model-type='cobweb' --seed=456
python3 main_mnist.py --experiment=0 --label=0 --model-type='cobweb' --seed=789
python3 main_mnist.py --experiment=0 --label=0 --model-type='cobweb' --seed=0
python3 main_mnist.py --experiment=0 --label=0 --model-type='cobweb' --seed=50

python3 main_mnist.py --experiment=0 --label=0 --model-type='fc' --seed=123
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc' --seed=456
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc' --seed=789
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc' --seed=0
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc' --seed=50

python3 main_mnist.py --experiment=0 --label=0 --model-type='fc-cnn' --seed=123
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc-cnn' --seed=456
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc-cnn' --seed=789
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc-cnn' --seed=0
python3 main_mnist.py --experiment=0 --label=0 --model-type='fc-cnn' --seed=50
