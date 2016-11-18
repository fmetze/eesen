export EESEN_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/tools/irstlm/bin/:$PWD:$PATH
export LD_LIBRARY_PATH=$EESEN_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
export LC_ALL=C

[ -f $EESEN_ROOT/tools/env.sh ] && . $EESEN_ROOT/tools/env.sh

if [[ `uname -n` =~ ip-* ]]; then
  # AWS instance
  export EESEN_ROOT=${HOME}/eesen
  export PATH=${PWD}/meine:${PWD}/utils:${HOME}/kenlm/bin:${EESEN_ROOT}/tools/sph2pipe_v2.5:${EESEN_ROOT}/tools/openfst/bin:${EESEN_ROOT}/src/featbin:${EESEN_ROOT}/src/decoderbin:${EESEN_ROOT}/src/fstbin:${EESEN_ROOT}/src/netbin:$PATH
  export TMPDIR=/dev/shm

elif [[ `uname -n` =~ comet-* ]]; then
    # SDSC Comet cluster
    export TMPDIR=/scratch/$USER/$SLURM_JOBID
    module load python
    module load gnu
    module load cuda

elif [[ `uname -n` =~ bridges ]]; then
    # PSC Bridges cluster
    [ -z "$SLURM_JOBID" ] && export TMPDIR=/tmp || export TMPDIR=/local/$SLURM_JOBID
    #export TMPDIR=$LOCAL
    #export TMPDIR=.
    export PATH=$PATH:/pylon2/ir3l68p/metze/sox-14.4.2/src

elif [[ `uname -n` =~ compute-* ]]; then
    # CMU Rocks cluster
    module load python27
    module load gcc-4.9.2
    export TMPDIR=/scratch
fi

# just in case
[ -d $TMPDIR ] || export TMPDIR=/tmp

if [[ ! -z ${acwt+x} ]]; then
    # let's assume we're decoding
    export PATH=$EESEN_ROOT/src-nogpu/netbin:$PATH
    echo "Preferring non-gpu netbin code"
fi
