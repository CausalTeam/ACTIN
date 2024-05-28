gamma=$1

python runnables/train.py  +dataset=cancer_sim  +backbone=cancer_lstm +backbone/hparams/cancer=${gamma}_lstm exp.seed=10 exp.logging=False
python runnables/train.py  +dataset=cancer_sim  +backbone=cancer_lstm +backbone/hparams/cancer=${gamma}_lstm exp.seed=101 exp.logging=False
python runnables/train.py  +dataset=cancer_sim  +backbone=cancer_lstm +backbone/hparams/cancer=${gamma}_lstm exp.seed=1010 exp.logging=False
python runnables/train.py  +dataset=cancer_sim  +backbone=cancer_lstm +backbone/hparams/cancer=${gamma}_lstm exp.seed=10101 exp.logging=False
python runnables/train.py  +dataset=cancer_sim  +backbone=cancer_lstm +backbone/hparams/cancer=${gamma}_lstm exp.seed=101010 exp.logging=False