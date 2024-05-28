gamma=$1

python runnables/train.py  +dataset=cancer_sim_random  +backbone=cancer +backbone/hparams/cancer=${gamma} exp.seed=10 exp.logging=False
python runnables/train.py  +dataset=cancer_sim_random  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=101 exp.logging=False
python runnables/train.py  +dataset=cancer_sim_random  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=1010 exp.logging=False
python runnables/train.py  +dataset=cancer_sim_random  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=10101 exp.logging=False
python runnables/train.py  +dataset=cancer_sim_random  +backbone/wt=cancer +backbone/hparams/cancer=${gamma} exp.seed=101010 exp.logging=False