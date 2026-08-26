#!bin/sh
L=2
lmax=4
mkdir -p "./data"
rm -f "./log.txt"

for i in {3..3}; do
  theta=$(printf "%1.2fpi" $(echo "scale=3; 0.05 * $i" | bc))
  echo "calculate L=${L}, THETA=${theta}"

  julia --project=.. ./run_bp_whip.jl \
  --lx $L --ly $L --lz $L --theta $theta \
  --lmax $lmax \
  --method all --maxiter 500 --tol 1e-8 \
  --damping 1.0 --pairnorm true > log.txt

  fname="CUBE_L${L}_THETA${theta}_LMAX${lmax}_ZZ.txt"
  echo "generate data file $fname"
  mv "./log.txt" "data/$fname"
done