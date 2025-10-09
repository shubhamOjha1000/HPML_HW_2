# ECE-GY 9143 - High Performance Machine Learning
# Homework Assignment 2

#### Author Shubham Ojha



## C2:
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 2 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

## C3:
#### num_workers 0
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 0 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

#### num_workers 4
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 4 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

#### num_workers 8
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 8 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

#### num_workers 12
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 12 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

#### num_workers 16
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 16 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

## C4:

#### num_workers 1
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 1 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

#### num_workers 0
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 0 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```

## C5:

#### GPU

```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 0 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cuda
```


#### CPU
```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 0 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cpu
```


## C6:

#### SGD

```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 4 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cuda
```

#### SGD with nesterov

```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 4 --optimizer sgd_nesterov --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cuda
```


#### Adagrad

```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 4 --optimizer adagrad --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cuda
```


#### Adadelta

```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 4 --optimizer adadelta --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cuda
```


#### Adam

```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 4 --optimizer adam --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model RESNET18 --device cuda
```


## C7:

```
python expt_main.py --num_epochs 5 --device cpu --batch_size 128 --num_workers 0 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4 --train_data_path 'Give ur training path' --model ResNet18_NoBN --device cuda
```

# Q3 
```
python expt_main.py --train_data_path 'Give ur training path' --Q3 True

```






