#!/bin/zsh
L=20
CHI=32
NSAMPLE=1

julia ./peps_sqr_whip_noise.jl \
  --L=$L \
  --chi=$CHI \
  --theta-min-pi=-0.5 \
  --theta-max-pi=0.0 \
  --ntheta=41 \
  --critical-window-pi=0.05 \
  --critical-step-pi=0.0025 \
  --sigmas-pi=0,0.05,0.1,0.15 \
  --samples=$NSAMPLE \
  --selector=explicit \
  --gates="17,11>17,12;9,4>10,4" \
  --edge-ops=X