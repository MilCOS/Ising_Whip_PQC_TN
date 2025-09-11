using JLD2


include("mps_funcs.jl")

const ROOT_PATH = "$(homedir())/Documents/QC_whip/code-1d/"

# System parameters
N = parse(Int, ARGS[1])
isopen = parse(Bool, ARGS[2])
bdc = isopen ? "obc" : "pbc"
use_nl_op = parse(Bool, ARGS[3]) # whether to use nonlocal gate
chi = 32
cutoff = 0.0
cut_bond = Int(N/2)

# initial state |++...+>
ket_es = mps_equal_superposition(N, isopen)
sites = siteinds(ket_es)

# 
THETAS = Array{Float64}([pi*2 * i/200 for i in 0:200])
SvNs = Array{Float64}([])
MaxDims = Array{Int64}([])

psi = deepcopy(ket_es)
for theta in THETAS
  gates = make_1d_whip_gates(N, theta, use_nl_op)
  zytensors = make_tensormap_of_ZY(gates, sites)
  psi .= ket_es
#   psi = deepcopy(ket_es)
  # make_contraction!(gates, zytensors, psi, chi, cutoff)
  make_contraction_buildin!(gates, zytensors, psi, chi, cutoff)
  s_vn = entanglement_entropy_svd(psi, cut_bond)
  
  push!(MaxDims, maxlinkdim(psi))
  push!(SvNs, s_vn)
end

println(maximum(MaxDims))
println(maximum(SvNs))

dir_str = "$(ROOT_PATH)/data/"
if !isdir("$dir_str")
    mkdir("$dir_str")
end
file_str = "1D_chain_Whip_$(bdc)_UseNlOp$(Int(use_nl_op))_N$(N)_Chi$(chi)"

jldsave("$dir_str$file_str"*".jld2"; 
    saved_svn=SvNs, 
    saved_linkdim=MaxDims,
    saved_thetas=THETAS
)
