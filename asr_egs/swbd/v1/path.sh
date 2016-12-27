export EESEN_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$PWD:$PATH
export LD_LIBRARY_PATH=$EESEN_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
export LC_ALL=C

[ -f $EESEN_ROOT/tools/env.sh ] && . $EESEN_ROOT/tools/env.sh

if [[ `uname -n` =~ comet-* ]]; then
    # SDSC Comet cluster
    module load python
    module load gnu
#    module load cuda
    [ -z "$SLURM_JOB_ID" ] && export TMPDIR=/tmp || export TMPDIR=/scratch/$USER/$SLURM_JOB_ID

elif [[ `uname -n` =~ ip-* ]]; then
    # AWS instance
    export EP=/home/ec2-user/eesen-precond
    export PATH=$PWD/meine:$PWD/utils:$EP/tools/openfst/bin:$EP/src/featbin:$EP/src/decoderbin:$EP/src/fstbin:$EP/src/netbin:$PATH

    export TMPDIR=/media/ephemeral0

elif [[ `uname -n` =~ bridges ]]; then
    # PSC Bridges cluster
    [ -n "$SLURM_JOBID" ] && export TMPDIR=/local/$SLURM_JOBID
    #[ -d $TMPDIR ] || export TMPDIR=/pylon1/ir3l68p/metze
    #export TMPDIR=$LOCAL
    #export TMPDIR=.
    export PATH=/pylon2/ir3l68p/metze/sox-14.4.2/src:${PATH}

elif [[ `uname -n` =~ compute-* ]]; then
    # CMU Rocks cluster
    module load python27
    module load gcc-4.9.2
    export TMPDIR=/scratch
fi

if [[ ! -z ${acwt+x} ]]; then
    # let's assume we're decoding
    export PATH=$EESEN_ROOT/src-nogpu/netbin:$PATH
    echo "Preferring non-gpu netbin code"
fi
