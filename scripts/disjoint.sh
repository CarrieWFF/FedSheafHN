python main_formal.py  --model 'fedsheaf'\
                --dataset 'ogbn-arxiv' \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-clients 10\
                --server-model 'DiagSheaf'\
                --client_train_epochs 10\
                --n_runs 1\
                --server_dropout 0.3\
                --client_dropout 0.6\
                --hn_dropout 0.3\
                --seed 435\
                --txt_no 15\
                --n-rnds 200\
                --gpu 0

