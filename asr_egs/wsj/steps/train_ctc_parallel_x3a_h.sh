#!/bin/bash

# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
# Copyright 2016  Florian Metze (Carnegie Mellon University)
# Apache 2.0

# This script trains acoustic models based on CTC and using SGD.

## Begin configuration section
train_tool=train-ctc-parallel  # the command for training; by default, we use the
                # parallel version which processes multiple utterances at the same time

# configs for multiple sequences
num_sequence=60          # during training, how many utterances to be processed in parallel
valid_num_sequence=60    # number of parallel sequences in validation
frame_num_limit=30000    # the number of frames to be processed at a time in training; this config acts to
         # to prevent running out of GPU memory if #num_sequence very long sequences are processed; the max
         # number of training examples is decided by if num_sequence or frame_num_limit is reached first.

# learning rate
learn_rate=4e-5          # learning rate
final_learn_rate=1e-6    # final learning rate
momentum=0.9             # momentum

# parallelization settings
nj=4                     # number of jobs in parallel
utts_per_avg=1000        # number of utterances per averaging step

# learning rate schedule
max_iters=25             # max number of iterations
min_iters=               # min number of iterations
start_epoch_num=1        # start from which epoch, used for resuming training from a break point

start_halving_inc=0.2    # start halving learning rates when the accuracy improvement falls below this amount
end_training_inc=0.02    # terminate training when the accuracy improvement falls below this amount
halving_factor=0.5       # learning rate decay factor
halving_after_epoch=10   # halving becomes enabled after this many epochs
force_halving_epoch=     # force halving after this epoch

# logging
report_step=5000         # during training, the step (number of utterances) of reporting objective and accuracy
verbose=1

# feature configs
sort_by_len=true         # whether to sort the utterances by their lengths
seed=777                 # random seed
block_softmax=false      # multi-lingual training

splice_feats=false       # whether to splice neighboring frams
subsample_feats=false    # whether to subsample features
norm_vars=true           # whether to apply variance normalization when we do cmn
add_deltas=true          # whether to add deltas
copy_feats=true          # whether to copy features into a local dir (on the GPU machine)

# status of learning rate schedule; useful when training is resumed from a break point
cvacc=0
pvacc=0
halving=false

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-tr> <data-cv> <exp-dir>"
   echo " e.g.: $0 data/train_tr data/train_cv exp/train_phn"
   exit 1;
fi

data_tr=$1
data_cv=$2
dir=$3

mkdir -p $dir/log $dir/nnet

for f in $data_tr/feats.scp $data_cv/feats.scp $dir/labels.tr.gz $dir/labels.cv.gz $dir/nnet.proto; do
  [ ! -f $f ] && echo `basename "$0"`": no such file $f" && exit 1;
done

## Read the training status for resuming
[ -f $dir/.epoch ]   && start_epoch_num=`cat $dir/.epoch 2>/dev/null`
[ -f $dir/.cvacc ]   && cvacc=`cat $dir/.cvacc 2>/dev/null`
[ -f $dir/.pvacc ]   && pvacc=`cat $dir/.pvacc 2>/dev/null`
[ -f $dir/.halving ] && halving=`cat $dir/.halving 2>/dev/null`
[ -f $dir/.lrate ]   && learn_rate=`cat $dir/.lrate 2>/dev/null`

## Set up labels
labels_tr="ark:gunzip -c $dir/labels.tr.gz|"
labels_cv="ark:gunzip -c $dir/labels.cv.gz|"
# Compute the occurrence counts of labels in the label sequences. These counts will be used to
# derive prior probabilities of the labels.
gunzip -c $dir/labels.tr.gz | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& $dir/log/compute_label_counts.log || exit 1
##

## Setup up features
# output feature configs which will be used in decoding
echo $norm_vars > $dir/norm_vars
echo $add_deltas > $dir/add_deltas
echo $splice_feats > $dir/splice_feats
echo $subsample_feats > $dir/subsample_feats

if $sort_by_len; then
  td=$(mktemp -d)
  feat-to-len scp:$data_tr/feats.scp ark,t:- | awk '{print $2}' > $td/len.tmp || exit 1;
  gzip -cd $dir/labels.tr.gz | paste -d" " $td/len.tmp $data_tr/feats.scp - | sort -gk 1 | \
      awk '{out=""; for (i=5;i<=NF;i++) {out=out" "$i}; if (!(out in done) && $1 > 3*NF) {done[out]=1; print $2 " " $3}}' > $dir/train.scp
  rm -rf $td
  feat-to-len scp:$data_cv/feats.scp ark,t:- | awk '{print $2}' | \
    paste -d " " $data_cv/feats.scp - | sort -k3 -n - | awk '{print $1 " " $2}' > $dir/cv.scp || exit 1;
else
  cat $data_tr/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
  cat $data_cv/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/cv.scp
fi

feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$dir/train.scp ark:- |"
feats_cv="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$dir/cv.scp ark:- |"

if $splice_feats; then
  feats_tr="$feats_tr splice-feats --left-context=1 --right-context=1 ark:- ark:- |"
  feats_cv="$feats_cv splice-feats --left-context=1 --right-context=1 ark:- ark:- |"
fi

if $subsample_feats; then
  tmpdir=$(mktemp -d)

  copy-feats "$feats_tr subsample-feats --n=3 --offset=0 ark:- ark:- |" \
             ark,scp:$tmpdir/train0.ark,$tmpdir/train0local.scp || exit 1;
  copy-feats "$feats_tr subsample-feats --n=3 --offset=1 ark:- ark:- |" \
             ark,scp:$tmpdir/train1.ark,$tmpdir/train1local.scp || exit 1;
  copy-feats "$feats_tr subsample-feats --n=3 --offset=2 ark:- ark:- |" \
             ark,scp:$tmpdir/train2.ark,$tmpdir/train2local.scp || exit 1;
  copy-feats "$feats_cv subsample-feats --n=3 --offset=0 ark:- ark:- |" \
             ark,scp:$tmpdir/cv0.ark,$tmpdir/cv0local.scp || exit 1;
  copy-feats "$feats_cv subsample-feats --n=3 --offset=1 ark:- ark:- |" \
             ark,scp:$tmpdir/cv1.ark,$tmpdir/cv1local.scp || exit 1;
  copy-feats "$feats_cv subsample-feats --n=3 --offset=2 ark:- ark:- |" \
             ark,scp:$tmpdir/cv2.ark,$tmpdir/cv2local.scp || exit 1;

  sed 's/^/0x/' $tmpdir/train0local.scp        > $tmpdir/train_local.scp
  sed 's/^/1x/' $tmpdir/train1local.scp | tac >> $tmpdir/train_local.scp
  sed 's/^/2x/' $tmpdir/train2local.scp       >> $tmpdir/train_local.scp
  sed 's/^/0x/' $tmpdir/cv0local.scp  > $tmpdir/cv_local.scp
  sed 's/^/1x/' $tmpdir/cv1local.scp >> $tmpdir/cv_local.scp
  sed 's/^/2x/' $tmpdir/cv2local.scp >> $tmpdir/cv_local.scp

  feats_tr="ark,s,cs:copy-feats scp:$tmpdir/feats_tr.JOB.scp ark:- |"
  feats_cv="ark,s,cs:copy-feats scp:$tmpdir/feats_cv.JOB.scp ark:- |"

  gzip -cd $dir/labels.tr.gz | sed 's/^/0x/'  > $tmpdir/labels.tr
  gzip -cd $dir/labels.cv.gz | sed 's/^/0x/'  > $tmpdir/labels.cv
  gzip -cd $dir/labels.tr.gz | sed 's/^/1x/' >> $tmpdir/labels.tr
  gzip -cd $dir/labels.cv.gz | sed 's/^/1x/' >> $tmpdir/labels.cv
  gzip -cd $dir/labels.tr.gz | sed 's/^/2x/' >> $tmpdir/labels.tr
  gzip -cd $dir/labels.cv.gz | sed 's/^/2x/' >> $tmpdir/labels.cv

  labels_tr="ark:cat $tmpdir/labels.tr|"
  labels_cv="ark:cat $tmpdir/labels.cv|"

  #trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT

elif $copy_feats; then
  # Save the features to a local dir on the GPU machine. On Linux, this usually points to /tmp
  tmpdir=$(mktemp -d)
  copy-feats "$feats_tr" ark,scp:$tmpdir/train.ark,$tmpdir/train_local.scp || exit 1;
  copy-feats "$feats_cv" ark,scp:$tmpdir/cv.ark,$tmpdir/cv_local.scp || exit 1;
  feats_tr="ark,s,cs:copy-feats scp:$tmpdir/feats_tr.JOB.scp ark:- |"
  feats_cv="ark,s,cs:copy-feats scp:$tmpdir/feats_cv.JOB.scp ark:- |"

  #trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT
fi

if $add_deltas; then
    feats_tr="$feats_tr add-deltas ark:- ark:- |"
    feats_cv="$feats_cv add-deltas ark:- ark:- |"
fi
## End of feature setup

# Initialize model parameters
if [ ! -f $dir/nnet/nnet.iter0 ]; then
    echo "Initializing model as $dir/nnet/nnet.iter0"
    net-initialize --binary=true --seed=$seed $dir/nnet.proto $dir/nnet/nnet.iter0 >& $dir/log/initialize_model.log || exit 1;
fi

# create another tmp directory for the averaging and shuffling operations
mkdir -p $tmpdir/avg $tmpdir/shuffle
cp $dir/nnet/nnet.iter$[start_epoch_num-1] $tmpdir/avg || exit 1;
cp $tmpdir/train_local.scp $tmpdir/train_local.org
cp $tmpdir/cv_local.scp    $tmpdir/cv_local.org

# main loop
cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"
echo "[NOTE] TOKEN_ACCURACY refers to token accuracy, i.e., (1.0 - token_error_rate)."
for iter in $(seq $start_epoch_num $max_iters); do
    hvacc=$pvacc
    pvacc=$cvacc

    # distribute and shuffle the data for this iteration
    utils/prep_scps.sh --nj $nj --cmd "run.pl" --seed $iter \
	$tmpdir/train_local.org $tmpdir/cv_local.org $num_sequence $frame_num_limit $tmpdir/shuffle $tmpdir >& \
        $dir/log/shuffle.iter$iter.log || exit 1;
    rm $tmpdir/batch.tr.list  $tmpdir/batch.cv.list
    
    # train
    echo -n "EPOCH $iter RUNNING ... "
    for JOB in `seq 1 $nj`; do
	F=`echo $feats_tr|awk -v j=$JOB '{ sub("JOB", j); print $0 }'`
	$train_tool --report-step=$report_step --num-sequence=$num_sequence --frame-limit=$frame_num_limit \
            --learn-rate=$learn_rate --momentum=$momentum --verbose=$verbose --block-softmax=$block_softmax \
            --num-jobs=$nj --job-id=$JOB \
	    "$F" "$labels_tr" $tmpdir/avg/nnet.iter$[iter-1] $tmpdir/avg/nnet.iter${iter} \
            >& $dir/log/tr.iter$iter.$JOB.log &
	sleep 15
    done
    wait
    cp $tmpdir/avg/nnet.iter$iter $dir/nnet
    end_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
    echo -n "ENDS [$end_time]: "
    tracc=$(grep -a "TOTAL TOKEN_ACCURACY" $dir/log/tr.iter${iter}.1.log | tail -n 1 | awk '{ acc=$4; gsub("%","",acc); print acc }')
    echo -n "lrate $(printf "%.6g" $learn_rate), TRAIN ACCURACY $(printf "%.4f" $tracc)%, "

    # validation
    for JOB in `seq 1 $nj`; do
	F=`echo $feats_cv|awk -v j=$JOB '{ sub("JOB", j); print $0 }'`
	$train_tool --report-step=$report_step --num-sequence=$valid_num_sequence --frame-limit=$frame_num_limit \
            --cross-validate=true --block-softmax=$block_softmax \
            --learn-rate=$learn_rate --momentum=$momentum --verbose=$verbose \
	    --num-jobs=$nj --job-id=$JOB \
            "$F" "$labels_cv" $tmpdir/avg/nnet.iter${iter} \
            >& $dir/log/cv.iter$iter.$JOB.log &
	sleep 15
    done
    wait
    cvacc=$(grep -a "TOTAL TOKEN_ACCURACY" $dir/log/cv.iter${iter}.1.log | tail -n 1 | awk '{ acc=$4; gsub("%","",acc); print acc }')
    echo "VALID ACCURACY $(printf "%.4f" $cvacc)%"

    # stopping criterion
    intre='^[0-9]+$'
    old_impr=$(bc <<< "($pvacc-$hvacc)")
    rel_impr=$(bc <<< "($cvacc-$pvacc)")
    if $halving && [ 1 == $(bc <<< "$rel_impr < $end_training_inc") ] \
                && [ 1 == $(bc <<< "$old_impr < $end_training_inc") ]; then
      if [[ ( "$min_iters" =~ $intre ) && ( $min_iters -gt $iter ) ]]; then
        echo we were supposed to finish, but we continue as min_iters : $min_iters
      else
        echo finished, too small rel. improvement $rel_impr and $old_impr
        break
      fi
    fi

    # start annealing when improvement is low
    if [ 1 == $(bc <<< "$rel_impr < $start_halving_inc") ] && [ $iter -ge $halving_after_epoch ] && \
       [ 1 == $(bc <<< "$old_impr < $start_halving_inc") ]; then halving=true; fi
    if [[ ( "$force_halving_epoch" =~ $intre ) && ( $iter -ge $force_halving_epoch ) ]]; then halving=true; fi

    # do annealing
    if $halving; then
      learn_rate=$(awk "BEGIN {print($learn_rate*$halving_factor)}")
      learn_rate=$(awk "BEGIN {if ($learn_rate<$final_learn_rate) {print $final_learn_rate} else {print $learn_rate}}")
    fi

    # save the status
    echo $[$iter+1] > $dir/.epoch    # +1 because we save the epoch to start from
    echo $cvacc > $dir/.cvacc
    echo $pvacc > $dir/.pvacc
    echo $halving > $dir/.halving
    echo $learn_rate > $dir/.lrate
done

# Convert the model marker from "<BiLstmParallel>" to "<BiLstm>"
format-to-nonparallel $dir/nnet/nnet.iter${iter} $dir/final.nnet >& $dir/log/model_to_nonparal.log || exit 1;

echo "Training succeeded. The final model $dir/final.nnet"
