#!/bin/bash

# Copyright 2016 Carnegie Mellon University (Author: Florian Metze)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Mix the contents of various data directories for data augmentation
# A provided speaker list is split, and then the data from various source dirs is combined


echo "$0 $@"  # Print the command line for logging


seed=$1            ;# random seed
indir=$2           ;# the directory from which to read the initial speaker list to split
targetdir=$3       ;# output directory
sources=( ${@:4} ) ;# the source directories, from which to take the data

mydir=`mktemp -d`
trap "rm -rf $mydir" EXIT

i=${#sources[@]}
for src in ${sources[@]}; do
    echo "*** now reading" $src "***"
    [ $i -eq 1 ] && break
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent `awk -v v=$i 'BEGIN {print int(100/v)}'` \
        --seed $seed $indir ${mydir}/tmp$i ${mydir}/tmp
    utils/subset_data_dir.sh --spk-list ${mydir}/tmp/spk2utt \
	$src ${mydir}/com$i
    indir=${mydir}/tmp$i
    let i=$i-1
done

utils/subset_data_dir.sh --spk-list ${mydir}/tmp2/spk2utt $src ${mydir}/com1
echo "*** now combining sub-sets ***"
utils/combine_data.sh --skip-fix true ${mydir}/all ${mydir}/com*
echo "*** now filtering for original utterances ***"
utils/subset_data_dir.sh --utt-list $2/utt2spk ${mydir}/all $targetdir
echo "*** result ***"
wc $targetdir/*
