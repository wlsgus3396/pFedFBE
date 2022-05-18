
python3 main.py --dataset 'Lasso' --model 'Lasso' --learning_rate 0.0005 --beta 1.0 --lamda 15 --num_global_iters 100 --algorithm 'FedDual' --personal_learning_rate 0.0005 --times 1 --gpu 0 --regularizer 'l1' --lamdaCO 1 --l1const 1100 --l2const 7

python3 main.py --dataset 'Lasso' --model 'Lasso' --learning_rate 0.0005 --beta 1.0 --lamda 15 --num_global_iters 100 --algorithm 'pSCAFFOLD' --personal_learning_rate 0.0005 --times 1 --gpu 0 --regularizer 'l1' --lamdaCO 1 --l1const 1100 --l2const 7