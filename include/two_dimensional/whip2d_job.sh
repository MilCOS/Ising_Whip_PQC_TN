#!/bin/zsh

# Ls=(2 4 6 8 10)
# chi=1024
# CHIs=(1024) #(4 8 16 32 64)

# Ls=(4 6 8 10 20 30 40 50 60 80 110 150 200)
# Ls=(30 40 50 60 80 110 150 200)
L=100
chi=32
CHIs=(4 8 12 16 20 24 28 36 40)
use_projector=false
cutoff=1E-16

export OMP_NUM_THREADS=2

# for chi in $CHIs; do
    print L $L chi $chi;
    # julia --sysimage ~/.julia/sysimages/sys_itensors.so ./mps_sqr_whip.jl $L $chi
    julia --sysimage ~/.julia/sysimages/sys_itensors.so ./peps_sqr_whip_scaling.jl $L $chi $use_projector $cutoff
    print "===== ====="
# done
