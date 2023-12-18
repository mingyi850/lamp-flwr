#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:3:$len}


init_len_values=(0.2 0.6 1.0 1.4 1.8 2.2 2.6 3.0)
    
for i in ${init_len_values[@]}:
do
    python3 attack.py --dataset $2 --split test --loss cos --n_inputs 100 -b $3 --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --bert_path $1 --n_steps 2000 --init_size $i  $last_args 
done