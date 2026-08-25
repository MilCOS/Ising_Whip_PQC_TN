#!bin/sh
L=4
mkdir -p "./data"

for i in {1..10}; do
  theta=$(printf "%1.2fpi" $(echo "scale=3; 0.05 * $i" | bc))
  echo "calculate l=${L}, theta=${theta}"

  julia --project=.. ./run_bp_whip.jl \
  --lx $L --ly $L --lz $L --theta $theta \
  --method all --maxiter 300 --tol 1e-9 \
  --damping 0.5 --pairnorm true > log.txt

  fname="CUBE_L${L}_THETA${theta}_ZZ.txt"
  echo "generate data file $fname"
  mv "./log.txt" "data/$fname"
done