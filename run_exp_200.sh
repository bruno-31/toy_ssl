path=./log_decoupled_50lbl
script=train_decoupled_cifar.py

python $script --logdir $path/g5e-3_e10_seed10 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False

python $script --logdir $path/vanilla_seed10 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False --vanilla True


python $script --logdir $path/g5e-3_e10_seed20 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False --seed 20

python $script --logdir $path/vanilla_seed20 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False --vanilla True --seed 20


python $script --logdir $path/g5e-3_e10_seed30 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False --seed 30

python $script --logdir $path/vanilla_seed30 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False --vanilla True --seed 30


python $script --logdir $path/g5e-3_e10_seed40 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False --seed 40

python $script --logdir $path/vanilla_seed40 --gamma 0.005 --epsilon 10. --eta 1. --data_dir /tmp/data/cifar-10-python --labeled 50 --epoch 800 --large False --vanilla True --seed 40


