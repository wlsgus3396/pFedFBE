python3 main.py --dataset 'Lasso' --model 'Lasso' --learning_rate 0.0005 --beta 1.0 --lamda 10000 --num_global_iters 100 --algorithm 'pFedFBE' --personal_learning_rate 0.0005 --times 1 --gpu 1 --regularizer 'l1' --lamdaCO 1 --l1const 1100 --l2const 7

python3 main.py --dataset 'Lasso' --model 'Lasso' --learning_rate 0.0005 --beta 1.0 --lamda 15 --num_global_iters 100 --algorithm 'pFedDyn' --personal_learning_rate 0.0005 --times 1 --gpu 1 --regularizer 'l1' --lamdaCO 1 --l1const 1100 --l2const 7

