#!/bin/bash

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=0
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=0
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=0
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=0
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=0

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=0

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=0

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=0

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=0
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=0

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=1
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=1
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=1
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=1
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=1

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=1

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=1

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=1

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=1
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=1

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=2
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=2
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=2
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=2
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=2

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=2

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=2

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=2

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=2
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=2

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=3
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=3
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=3
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=3
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=3

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=3

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=3

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=3

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=3
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=3

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=4
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=4
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=4
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=4
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=4

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=4

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=4

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=4

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=4
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=4

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=5
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=5
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=5
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=5
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=5

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=5

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=5

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=5

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=5
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=5

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=6
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=6
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=6
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=6
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=6

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=6

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=6

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=6

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=6
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=6

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=7
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=7
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=7
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=7
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=7

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=7

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=7

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=7

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=7
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=7

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=8
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=8
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=8
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=8
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=8

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=8

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=8

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=8

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=8
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=8

# --------------------------

python3 main.py --experiment=1 --model-type=cobweb --seed=123 --test=chosen --n-split=10 --label=9
python3 main.py --experiment=1 --model-type=cobweb --seed=456 --test=chosen --n-split=10 --label=9
python3 main.py --experiment=1 --model-type=cobweb --seed=789 --test=chosen --n-split=10 --label=9
python3 main.py --experiment=1 --model-type=cobweb --seed=0 --test=chosen --n-split=10 --label=9
python3 main.py --experiment=1 --model-type=cobweb --seed=50 --test=chosen --n-split=10 --label=9

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=9

python3 main.py --experiment=1 --model-type=fc --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=9

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=fast --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=fast --label=9

python3 main.py --experiment=1 --model-type=fc-cnn --seed=123 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=456 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=789 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=0 --test=chosen --n-split=10 --nn-ver=slow --label=9
python3 main.py --experiment=1 --model-type=fc-cnn --seed=50 --test=chosen --n-split=10 --nn-ver=slow --label=9
