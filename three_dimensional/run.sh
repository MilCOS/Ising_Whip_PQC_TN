#!bin/sh
L=6
lmax=4
mkdir -p "./data"
rm -f "./log.txt"

for i in {0..9}; do
  theta=$(printf "%1.3fpi" $(echo "scale=4; 0.025 + 0.05 * $i" | bc))
  echo "calculate L=${L}, THETA=${theta} with LMAX=${lmax}"

  julia --project=.. ./run_bp_whip.jl \
  --lx $L --ly $L --lz $L --theta $theta \
  --lmax $lmax \
  --method all --maxiter 1000 --tol 1e-8 \
  --damping 0.5 --pairnorm true > log_${L}_${lmax}.txt

  fname="CUBE_L${L}_THETA${theta}_LMAX${lmax}_ZZ.txt"
  echo "generate data file ${fname}"
  mv "./log_${L}_${lmax}.txt" "data/${fname}"
done