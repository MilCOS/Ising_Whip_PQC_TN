#!bin/sh
L=2
lmax=6
mkdir -p "./data"
rm -f "./log.txt"

for i in {1..10}; do
  theta=$(printf "%1.2fpi" $(echo "scale=3; 0.05 * $i" | bc))
  echo "calculate L=${L}, THETA=${theta}"

  julia --project=.. ./run_bp_whip.jl \
  --lx $L --ly $L --lz $L --theta $theta \
  --lmax $lmax \
  --method all --maxiter 1000 --tol 1e-8 \
  --damping 0.5 --pairnorm true > log_${lmax}.txt

  fname="CUBE_L${L}_THETA${theta}_LMAX${lmax}_ZZ.txt"
  echo "generate data file ${fname}"
  mv "./log_${lmax}.txt" "data/${fname}"
done