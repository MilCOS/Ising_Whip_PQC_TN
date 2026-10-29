using JLD2, Printf

include("peps_funcs.jl")

const ROOT_PATH = "$(homedir())/Documents/QC_whip/code-2d/"

# System parameters
L = parse(Int, ARGS[1])
chi = parse(Int, ARGS[2])
global min_chi = 4
cutoff = parse(Float64, ARGS[3])
N = L*L

# initial state |++...+>
ket_es = twodim_sqr_peps(L, "+"^N);
sites = get_siteinds(ket_es)

# THETAS = Array{Float64}([pi/4*i/50 for i in 0:1:10]) .+ pi/4
THETAS = Array{Float64}([pi/2*i/20 for i in -40:1:40])
println(THETAS./pi)
LinkDims = Array{Vector{Int64}}([])
# pair index
p_indices = Tuple[]
for y in L-1:-1:1
  # diagonal
  i0, i1 = coor2idx((L,L), L), coor2idx((y,y), L)
  push!(p_indices, (i0,i1))
  # edges
  i0, i1 = coor2idx((L,y), L), coor2idx((y,L), L)
  push!(p_indices, (i0,i1))
end
p_indices = [(L*L, L*L-1)] # skip pair
println(p_indices)
ZdZ0s = Array{Vector{Float64}}([])
# edge index
# b_indices = [coor2idx((c,1),L) for c in 2:L-1]
b_indices = [coor2idx((L,c),L) for c in 1:L-1]
b_indices = [b_indices; [coor2idx((L-c,L),L) for c in 1:L-1]]
# b_indices = [b_indices; [coor2idx((1,L-c),L) for c in 1:L-2]]
b_indices = stack(b_indices)
b_indices = transpose(b_indices)
b_indices = collect(Iterators.flatten(b_indices))
# unique!(b_indices)
# b_indices = [2] # skip edge
println(b_indices)
Xs = Array{Vector{Float64}}([])


println("Start iteration...")
start_time = time_ns()
for (i,theta) in enumerate(THETAS)
  println("$(theta/pi)")
  psi = deepcopy(ket_es)
  # run the whip circuit
  gates = make_sqr_whip_gates_obc(L, theta)
  zytensors = make_tensormap_of_ZY(gates, sites)
  make_contraction_peps!(gates, zytensors, psi, chi, cutoff)
  # zytensors = make_mpomap_of_ZY(gates, sites, "LR")
  # make_mpo_contraction_peps!(gates, zytensors, psi)

  # contract to obtain the normalization factor
  # val3 = contract_twopeps_naive(psi, psi, chi, cutoff, plev=0)
  val3 = contract_twopeps_bdr_mps(psi, psi, chi, cutoff, plev=0)
  # observable
  val_corr = Float64[]
  link_dims = nothing
  for (i0,i1) in p_indices
    zzpsi = deepcopy(psi)
    zzpsi[i0] *= op("Z", sites[i0])
    noprime!(zzpsi[i0], tags="Site")
    zzpsi[i1] *= op("Z", sites[i1])
    noprime!(zzpsi[i1], tags="Site")
    # zz_val, linkdims = contract_twopeps_naive(zzpsi, psi, chi, cutoff, plev=1)
    zz_val, _, link_dims = contract_twopeps_bdr_mps(zzpsi, psi, chi, cutoff, plev=1)
    push!(val_corr, real(zz_val/val3))
  end
  val_boundary = Float64[]
  for i0 in b_indices
    x_psi = deepcopy(psi)
    x_psi[i0] *= op("X", sites[i0])
    noprime!(x_psi[i0], tags="Site")
    x_val, _, linkdims = contract_twopeps_bdr_mps(x_psi, psi, chi, cutoff, plev=1)
    push!(val_boundary, real(x_val/val3))
  end
  push!(LinkDims, link_dims)
  push!(ZdZ0s, val_corr)
  push!(Xs, val_boundary)
end
# Zs = stack(Zs)
elapsed_ns = time_ns() - start_time

dir_str = "$(ROOT_PATH)/data/scaling"
if !isdir("$dir_str")
    mkdir("$dir_str")
end

if (length(b_indices) == 1) && (length(p_indices) != 1) # pair_
  file_str = "2D_sqr_PEPS_Whip_MPO_L$(L)_Chi$(chi)_ZZ_$(cutoff)_fss"
else # edge
  file_str = "2D_sqr_PEPS_wrong_Whip_MPO_L$(L)_Chi$(chi)_ZZ_$(cutoff)_half_spp"
end

jldsave("$dir_str/$file_str"*".jld2"; 
    saved_linkdim=LinkDims,
    saved_thetas=THETAS,
    saved_zdz0=ZdZ0s,
    saved_x0=Xs,
    total_time_sec=elapsed_ns * 1e-9
)