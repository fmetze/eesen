#!/bin/bash

### for CMU rocks cluster
#PBS -q gpu
#PBS -j oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00

### for XSede comet cluster ###
### submit sbatch --ignore-pbs train-2-gpu.sh
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --output=log/slurm-%j.out
#SBATCH --export=ALL
#SBATCH --time="48:00:00"

uname -a
date

. ./cmd.sh
. ./path.sh

[ -f local.conf ] && . ./local.conf

# Specify network structure and generate the network topology
context=1           # number of frames/ deltas we combine together
lstm_layer_num=5    # number of LSTM layers
lstm_cell_dim=320   # number of memory cells in every LSTM layer
num_seq=200         # the number of sequences to process in parallel
frame_limit=25000   # the maximum number of frames to process per mini-batch
input_dim=80        # project input down to
proj_dim=320        # dimensionality of projection layer

# Need to set a variable before we can change it here!
. ./utils/parse_options.sh

# Set the experiment directory
dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}_n${num_seq}_f${frame_limit}_aux25
mkdir -p $dir

# Output the network topology
utils/model_topo.py --lstm-layer-num ${lstm_layer_num} --lstm-cell-dim ${lstm_cell_dim} \
    --fgate-bias-init 1.0 \
    --input-feat-dim `feat-to-dim scp:'head -n 1 data/train_nodup/feats.scp|' ark,t:|awk -v c=$context '{print (1+2*c)*$2}'` \
    --target-num `awk 'END {print 1+$2}' data/lang_phn/units.txt` > $dir/nnet.proto
utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_nodup/text "<UNK>" | gzip -c - > $dir/labels.tr.gz
utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_dev/text   "<UNK>" | gzip -c - > $dir/labels.cv.gz

traindata=data/train_nodup
a=1 && [ -f ${dir}/.epoch ] && a=`cat ${dir}/.epoch`
mytmp=`mktemp -d`
trap "echo \"Removing features tmpdir $mytmp @ $(hostname)\"; rm -r $mytmp" EXIT

# Train the network with CTC. Refer to the script for details about the arguments
for m in $(seq $a 32); do
    [ -d $traindata ] || utils/mix_data_dirs.sh $m data/train_nodup $traindata \
        data/train data/train_A* >& $dir/log/mix.iter${m}.log

    steps/train_ctc_parallel_x3c.sh \
        --add-deltas false --num-sequence $num_seq --frame-num-limit $frame_limit \
        --splice-feats true --subsample-feats true --max-iters $m \
        $traindata data/train_dev $dir || exit 1;

    # we want the augmented training data private to this process
    traindata=${mytmp}/X$m
done

echo Ok.
