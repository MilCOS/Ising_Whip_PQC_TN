#!/bin/zsh
#Ns=(4 6 8 10 12 14 16 18 20 30 40 50 60 70 80 100 200 300 400 500)
isopen=true
dN=20
use_nl_op=true

export OMP_NUM_THREADS=2

# for i in {1..20}; do
for ((i = 10; i <= 10; i++)); do
    N=$((i * dN))
    print "N $N, isopen $isopen"
    julia --sysimage ~/.julia/sysimages/sys_itensors.so ./mps_1d_whip.jl $N $isopen $use_nl_op
    print "===== ====="
done
