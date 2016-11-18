#!/bin/bash

. path.sh

if [ $# != 2 ] && [ $# != 8 ]; then
   echo "Usage: hkust_data_prep.sh [--audio-filter '| sox ...' --train-dir X --dev-dir Y] AUDIO_PATH TEXT_PATH"
   exit 1;
fi

if [ $# == 2 ]; then
    HKUST_AUDIO_DIR=$1
    HKUST_TEXT_DIR=$2
else
    HKUST_AUDIO_DIR=$7
    HKUST_TEXT_DIR=$8
fi

train_dir=data/train
dev_dir=data/dev
audio_filter=""

. parse_options.sh

case 0 in    #goto here
    1)
;;           #here:
esac

mkdir -p $train_dir data/local/train
mkdir -p $dev_dir   data/local/dev

#data directory check
if [ ! -d $HKUST_AUDIO_DIR ] || [ ! -d $HKUST_TEXT_DIR ]; then
  echo "Error: run.sh requires two directory arguments"
  exit 1;
fi

#find sph audio file for train dev resp.
find $HKUST_AUDIO_DIR -iname "*.sph" | grep -i "audio/train" > data/local/train/sph.flist
find $HKUST_AUDIO_DIR -iname "*.sph" | grep -i "audio/dev" > data/local/dev/sph.flist

n=`cat data/local/train/sph.flist data/local/dev/sph.flist | wc -l`
[ $n -ne 897 ] && \
  echo Warning: expected 897 data data files, found $n


#Transcriptions preparation

[ -f data/local/train/transcripts.txt ] || \
#collect all trans, convert encodings to utf-8,
find $HKUST_TEXT_DIR -iname "*.txt" | grep -i "trans/train" | xargs cat |\
  iconv -f GBK -t utf-8 - | perl -e '
    while (<STDIN>) {
      @A = split(" ", $_);
      if (@A <= 1) { next; }
      if ($A[0] eq "#") { $utt_id = $A[1]; } 
      if (@A >= 3) {
        $A[2] =~ s:^([AB])\:$:$1:; 
        printf "%s-%s-%06.0f-%06.0f", $utt_id, $A[2], 100*$A[0] + 0.5, 100*$A[1] + 0.5; 
        for($n = 3; $n < @A; $n++) { print " $A[$n]" }; 
        print "\n"; 
      }
    }
  ' | sort -k1 > data/local/train/transcripts.txt 

[ -f data/local/dev/transcripts.txt ] || \
find $HKUST_TEXT_DIR -iname "*.txt" | grep -i "trans/dev" | xargs cat |\
  iconv -f GBK -t utf-8 - | perl -e '
    while (<STDIN>) {
      @A = split(" ", $_);
      if (@A <= 1) { next; }
      if ($A[0] eq "#") { $utt_id = $A[1]; } 
      if (@A >= 3) {
        $A[2] =~ s:^([AB])\:$:$1:; 
        printf "%s-%s-%06.0f-%06.0f", $utt_id, $A[2], 100*$A[0] + 0.5, 100*$A[1] + 0.5; 
        for($n = 3; $n < @A; $n++) { print " $A[$n]" }; 
        print "\n"; 
      }
    }
  ' | sort -k1  > data/local/dev/transcripts.txt



#transcripts normalization and segmentation 
#(this needs external tools),
#Download and configure segment tools  
pyver=`python --version 2>&1 | sed -e 's:^[A-Za-z ]*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/tools/mmseg-1.3.0/lib/python${pyver}/site-packages
if [ ! -f tools/mmseg-1.3.0/lib/python${pyver}/site-packages/*/mmseg.py ]; then
  echo "--- Downloading mmseg-1.3.0 ..."
  echo "NOTE: it assumes that you have Python, Setuptools installed on your system!"
  wget -P tools -q http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz 
  tar xf tools/mmseg-1.3.0.tar.gz -C tools
  cd tools/mmseg-1.3.0
  mkdir -p lib/python${pyver}/site-packages
  #ln -s lib lib64
  python setup.py build 
  python setup.py install --prefix=.
  cd ../..
  if [ ! -f tools/mmseg-1.3.0/lib/python${pyver}/site-packages/*/mmseg.py ]; then
    echo "mmseg is not found - installation failed?"
    exit 1
  fi
fi

[ -f data/local/train/text ] || \
cat data/local/train/transcripts.txt |\
  sed -e 's/<foreign language=\"[a-zA-Z]\+\">/ /g' |\
  sed -e 's/<\/foreign>/ /g' |\
  sed -e 's/<noise>\(.\+\)<\/noise>/\1/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  local/hkust_normalize.pl |\
  python local/hkust_segment.py |\
  awk '{if (NF > 1) print $0;}' > data/local/train/text

[ -f data/local/dev/text ] || \
cat data/local/dev/transcripts.txt |\
  sed -e 's/<foreign language=\"[a-zA-Z]\+\">/ /g' |\
  sed -e 's/<\/foreign>/ /g' |\
  sed -e 's/<noise>\(.\+\)<\/noise>/\1/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  local/hkust_normalize.pl |\
  python local/hkust_segment.py |\
  awk '{if (NF > 1) print $0;}' > data/local/dev/text

# some data is corrupted. Delete them
cat data/local/train/text | grep -v 20040527_210939_A901153_B901154-A-035691-035691 | egrep -v "A:|B:" > tmp
mv tmp data/local/train/text

#Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#sw02001-A_000098-001156 sw02001-A 0.98 11.56


awk '{ segment=$1; split(segment,S,"-"); side=S[2]; audioname=S[1];startf=S[3];endf=S[4];
   print segment " " audioname "-" side " " startf/100 " " endf/100}' < data/local/train/text > data/local/train/segments
awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' data/local/train/sph.flist > data/local/train/sph.scp

awk '{ segment=$1; split(segment,S,"-"); side=S[2]; audioname=S[1];startf=S[3];endf=S[4];
   print segment " " audioname "-" side " " startf/100 " " endf/100}' < data/local/dev/text > data/local/dev/segments
awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' data/local/dev/sph.flist > data/local/dev/sph.scp



sph2pipe=$EESEN_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -f $sph2pipe ] && echo "Could not find the sph2pipe program at $sph2pipe" && exit 1;

cat data/local/train/sph.scp | awk -v af="$audio_filter" -v sph2pipe="$sph2pipe" '{printf("%s-A %s -f wav -p -c 1 %s %s |\n", $1, sph2pipe, $2, af); 
    printf("%s-B %s -f wav -p -c 2 %s %s |\n", $1, sph2pipe, $2, af);}' | \
   sort > data/local/train/wav.scp || exit 1;

cat data/local/dev/sph.scp | awk -v af="$audio_filter" -v sph2pipe="$sph2pipe" '{printf("%s-A %s -f wav -p -c 1 %s %s |\n", $1, sph2pipe, $2, af); 
    printf("%s-B %s -f wav -p -c 2 %s %s |\n", $1, sph2pipe, $2, af);}' | \
   sort > data/local/dev/wav.scp || exit 1;
#side A - channel 1, side B - channel 2

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.
cat data/local/train/wav.scp | awk '{print $1}' | \
  perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_"; print "$1-$2 $1 $2\n"; ' \
  > data/local/train/reco2file_and_channel || exit 1;
cat data/local/dev/wav.scp | awk '{print $1}' | \
  perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_"; print "$1-$2 $1 $2\n"; ' \
  > data/local/dev/reco2file_and_channel || exit 1;


cat data/local/train/segments | awk '{spk=substr($1,1,33); print $1 " " spk}' > data/local/train/utt2spk || exit 1;
cat data/local/train/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > data/local/train/spk2utt || exit 1;

cat data/local/dev/segments | awk '{spk=substr($1,1,33); print $1 " " spk}' > data/local/dev/utt2spk || exit 1;
cat data/local/dev/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > data/local/dev/spk2utt || exit 1;

mkdir -p data/train data/dev
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp data/local/train/$f $train_dir/$f || exit 1;
done

for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp data/local/dev/$f $dev_dir/$f || exit 1;
done

echo HKUST data preparation succeeded
