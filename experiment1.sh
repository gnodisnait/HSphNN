#!/bin/sh

for IL in  0
do
for DIM in 2 3 15 30 100 200 1000 2000 3000
do
  python run_3.py  --valid  --init_loc ${IL}  --dim ${DIM} --pre_train 0 --ref_R 1 --data_dir data/ValidSyllogism/ --save --log DIM${DIM}R1InitLoc${IL}
done
done

python eval_exp1.py