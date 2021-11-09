## prepare data

(1)run *dataset_split.py* to generate train, valid, test sets
(2)run *list_generator.py* 3 times to generate lists contain train, valid, test samples
(3)run 
```
*python UCF2lmdb.py -f /home/yanlq/datasets/UCF101 --split train --size 41* 
*python UCF2lmdb.py -f /home/yanlq/datasets/UCF101 --split valid --size 7* 
*python UCF2lmdb.py -f /home/yanlq/datasets/UCF101 --split test  --size 12* 
```
generate video database for easily data retrive.

