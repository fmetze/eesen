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
#SBATCH --time="24:00:00"

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
dir=exp/train_char_l${lstm_layer_num}_c${lstm_cell_dim}-${proj_dim}_n${num_seq}_f${frame_limit}_aux22
mkdir -p $dir

# Output the network topology
utils/model_topo.py --lstm-layer-num ${lstm_layer_num} --lstm-cell-dim ${lstm_cell_dim} \
    --projection-dim ${proj_dim} --fgate-bias-init 1.0 \
    --input-feat-dim `feat-to-dim scp:'head -n 1 data/train_nodup/feats.scp|' ark,t:|awk -v c=$context '{print (1+2*c)*$2}'` \
    --target-num `awk 'END {print 1+$2}' data/lang_phn/units.txt` > $dir/nnet.proto
utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_nodup/text "<UNK>" | gzip -c - > $dir/labels.tr.gz
utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_nodup/text "<UNK>" | gzip -c - > $dir/labels.cv.gz

# Train the network with CTC. Refer to the script for details about the arguments
a=0 && [ -f ${dir}/.epoch ] && a=`cat ${dir}/.epoch | awk '{print $0-1}'`
mydir=`mktemp -d`
trap "echo \"Removing features tmpdir $mydir @ $(hostname)\"; rm -r $mydir" EXIT
for m in $(seq $a 32); do
    if [ -d $mydir/X$m ]; then
	steps/train_ctc_parallel_x3c.sh \
	    --add-deltas false --num-sequence $num_seq --frame-num-limit $frame_limit \
	    --splice-feats true --subsample-feats true --max-iters $m \
	    $mydir/X$m data/train_dev $dir || exit 1;
    fi

    # split training portion of data ('indir') n-fold using speakers,
    # and select accordingly from augmented copies (in target)
    # there is some naming problem ... train_1_pitch train_pitch_1
    source=data/train      ;# the data
    indir=data/train_nodup ;# the directory from which to read the speaker list etc
    n=5
    for i in $(seq $n -1 2); do
	utils/subset_data_dir_tr_cv.sh --cv-spk-percent `awk -v v=$i 'BEGIN {print int(100/v)}'` \
	    --seed $m $indir ${mydir}/tmp$i ${mydir}/tmp
	utils/subset_data_dir.sh --spk-list ${mydir}/tmp/spk2utt \
	    ${source}_`awk -v i=$i 'BEGIN {print i-1}'` ${mydir}/com$i
	indir=${mydir}/tmp$i
    done;
    utils/subset_data_dir.sh --spk-list ${mydir}/tmp2/spk2utt ${source} ${mydir}/com1
    utils/combine_data.sh ${mydir}/X`awk -v m=$m 'BEGIN {print m+1}'` ${mydir}/com*
    rm -rf ${mydir}/tmp* ${mydir}/com*
done
