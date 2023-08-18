#!/bin/bash

python3 main.py --experiment=0 --model-type=cobweb --seed=123 --test=entire --n-split=100
python3 main.py --experiment=0 --model-type=cobweb --seed=456 --test=entire --n-split=100
python3 main.py --experiment=0 --model-type=cobweb --seed=789 --test=entire --n-split=100
python3 main.py --experiment=0 --model-type=cobweb --seed=0 --test=entire --n-split=100
python3 main.py --experiment=0 --model-type=cobweb --seed=50 --test=entire --n-split=100

python3 main.py --experiment=0 --model-type=fc --seed=123 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc --seed=456 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc --seed=789 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc --seed=0 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc --seed=50 --test=entire --n-split=100 --nn-ver=fast

python3 main.py --experiment=0 --model-type=fc --seed=123 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc --seed=456 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc --seed=789 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc --seed=0 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc --seed=50 --test=entire --n-split=100 --nn-ver=slow

python3 main.py --experiment=0 --model-type=fc-cnn --seed=123 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc-cnn --seed=456 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc-cnn --seed=789 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc-cnn --seed=0 --test=entire --n-split=100 --nn-ver=fast
python3 main.py --experiment=0 --model-type=fc-cnn --seed=50 --test=entire --n-split=100 --nn-ver=fast

python3 main.py --experiment=0 --model-type=fc-cnn --seed=123 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc-cnn --seed=456 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc-cnn --seed=789 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc-cnn --seed=0 --test=entire --n-split=100 --nn-ver=slow
python3 main.py --experiment=0 --model-type=fc-cnn --seed=50 --test=entire --n-split=100 --nn-ver=slow

