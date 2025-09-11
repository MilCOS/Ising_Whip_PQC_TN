using JLD2
using Profile, PProf

include("mps_funcs.jl")

const ROOT_PATH = "$(homedir())/Documents/QC_whip/code-2d/"

# System parameters
L = parse(Int, ARGS[1])
chi = parse(Int, ARGS[2])
cutoff = 1e-8
N = L*L
cut_bond = Int(ceil(N/2))

# initial state |++...+>
ket_es = mps_equal_superposition(N)
sites = siteinds(ket_es)

# 
# if N<=50
THETAS = Array{Float64}([pi/4 * i/100 for i in 100:1:101])
# else
#   THETAS = Array{Float64}([pi/2 * i/N for i in -N:1:N])
# end
SvNs = Array{Float64}([])
EvNs = Array{Vector{Float64}}([])
MaxDims = Array{Int64}([])
ZZs = Array{Float64}([])

checklinkdim = true
psi = deepcopy(ket_es)
# szz = op("Sz", sites[N-1]) * op("Sz", sites[N])
i0, i1 = (L-1)*L+1, (L-1)*L+2
szz = op("Sz", sites[i0]) * op("Sz", sites[i1])
for theta in THETAS
  gates = make_sqr_whip_gates_obc(L, theta)
  zytensors = make_tensormap_of_ZY(gates, sites)
  psi .= ket_es
  make_contraction!(gates, zytensors, psi, chi, cutoff, checklinkdim)
  continue
  normalize!(psi)
  # make_contraction_buildin!(gates, zytensors, psi, chi, cutoff, checklinkdim)
  # expectation value
  zz_val = inner(psi, apply([szz], psi))
  # entanglement entropy
  s_vn, e_vn = entanglement_entropy_svd(psi, cut_bond)
  # configuration
  if (L <= 4) & (abs(theta-pi/2)<1e-16)
    s_conf = check_ground_state_configuration(psi)
    println(s_conf)
  end
  # store data
  push!(MaxDims, maxlinkdim(psi))
  push!(SvNs, s_vn)
  push!(EvNs, e_vn)
  push!(ZZs, zz_val)
end

println(maximum(MaxDims))
println(maximum(SvNs))
EvNs_new = zeros(Float64, length(THETAS), maximum(MaxDims)) .+ 100
for i in eachindex(THETAS)
  EvNs_new[i,1:length(EvNs[i])] = EvNs[i]
end

dir_str = "$(ROOT_PATH)/data/"
if !isdir("$dir_str")
    mkdir("$dir_str")
end
file_str = "2D_sqr_Whip_L$(L)_Chi$(chi)_ZZ"

# jldsave("$dir_str$file_str"*".jld2"; 
#     saved_svn=SvNs, 
#     saved_evn=EvNs_new,
#     saved_linkdim=MaxDims,
#     saved_thetas=THETAS,
#     saved_zz=ZZs
# )