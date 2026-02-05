using JLD2, Printf

include("peps_funcs.jl")

const ROOT_PATH = "$(homedir())/Documents/QC_whip/code-2d/"

# System parameters
L = parse(Int, ARGS[1])
chi = parse(Int, ARGS[2])
global min_chi = 4
use_projector = parse(Bool, ARGS[3])
cutoff = parse(Float64, ARGS[4])
N = L*L

# initial state |++...+>
ket_es = twodim_sqr_peps(L, "+"^N);
sites = get_siteinds(ket_es)

THETAS = Array{Float64}([pi/4*i/100 for i in -10:1:10]) .+ pi/4
# THETAS = Array{Float64}([pi/4*18/20,pi/4*19/20])#([pi/4*i/20 for i in -40:1:40])
LinkDims = Array{Vector{Int64}}([])
ZZs = Array{Float64}([])

# boundary index
b_indices = [[coor2idx((c,1),L), coor2idx((c,L),L), coor2idx((1,c),L), coor2idx((L,c),L)] for c in 1:L]
b_indices = stack(b_indices)
b_indices = transpose(b_indices)
b_indices = collect(Iterators.flatten(b_indices))
unique!(b_indices)
X_Xs = Array{Float64}([])

# pinning field
Tproj = op("Z",sites,1)
Tproj[sites[1]'=>1,sites[1]=>1] = 1.0
Tproj[sites[1]'=>2,sites[1]=>2] = 0.0
# observable at each site
Zs = []

println("Start iteration...")
start_time = time_ns()
for (i,theta) in enumerate(THETAS)
  psi = deepcopy(ket_es)
  # run the whip circuit
  gates = make_sqr_whip_gates_obc(L, theta)
  zytensors = make_tensormap_of_ZY(gates, sites)
  make_contraction_peps!(gates, zytensors, psi, chi, cutoff)
  # zytensors = make_mpomap_of_ZY(gates, sites, "LR")
  # make_mpo_contraction_peps!(gates, zytensors, psi)

  if use_projector
    psi[1] = psi[1] * Tproj
    noprime!(psi[1], tags="Site")
  end
  # contract to obtain the normalization factor
  # val3 = contract_twopeps_naive(psi, psi, chi, cutoff, plev=0)
  val3 = contract_twopeps_bdr_mps(psi, psi, chi, cutoff, plev=0)
  # observable
  val_tot = 0.0
  link_dims = nothing
  for (i0,i1) in [(L*L, L*L-1)] #gates.indices
    zzpsi = deepcopy(psi)
    zzpsi[i0] *= op("Z", sites[i0])
    noprime!(zzpsi[i0], tags="Site")
    zzpsi[i1] *= op("Z", sites[i1])
    noprime!(zzpsi[i1], tags="Site")
    # zz_val, linkdims = contract_twopeps_naive(zzpsi, psi, chi, cutoff, plev=1)
    zz_val, coeff, link_dims = contract_twopeps_bdr_mps(zzpsi, psi, chi, cutoff, plev=1)
    val_tot += real(zz_val/val3)
  end
  push!(LinkDims, link_dims)
  push!(ZZs, val_tot/N)
  # boundary X
  """
  x_xpsi = deepcopy(psi)
  for i0 in b_indices
    x_xpsi[i0] *= op("Z", sites[i0])
    noprime!(x_xpsi[i0], tags="Site")
  end
  x_x_val, _, linkdims = contract_twopeps_bdr_mps(x_xpsi, psi, chi, cutoff, plev=1)
  val_boundary = real(x_x_val/val3)
  push!(X_Xs, val_boundary)
  # every site Z
  ztmp = []
  for i1 in 1:N
    i0 = 1
    zpsi = deepcopy(psi)
    zpsi[i0] *= op("Z", sites[i0])
    noprime!(zpsi[i0], tags="Site")
    zpsi[i1] *= op("Z", sites[i1])
    noprime!(zpsi[i1], tags="Site")
    z_val, _, _ = contract_twopeps_bdr_mps(zpsi, psi, chi, cutoff, plev=1)
    push!(ztmp, real(z_val/val3))
  end
  push!(Zs, ztmp)
  if i%2==1
      @printf "theta = %1.3f pi, <Z_i Z_j> = %1.4f %.3f %i\n" (theta/pi) (val_tot) (val_boundary) maximum(linkdims)
  end
  """
  if i%2==1
      @printf "theta = %1.3f pi, <Z_i Z_j> = %1.4f %i\n" (theta/pi) (val_tot) maximum(link_dims)
  end
end
# Zs = stack(Zs)
elapsed_ns = time_ns() - start_time


dir_str = "$(ROOT_PATH)/data/scaling"
if !isdir("$dir_str")
    mkdir("$dir_str")
end

# file_str = "2D_sqr_PEPS_Whip_MPO_L$(L)_Chi$(chi)_ZZ_Z_P$(use_projector)"
if min_chi == chi
  file_str = "2D_sqr_PEPS_Whip_MPO_L$(L)_Chi$(chi)_ZZ_$(min_chi)_$(cutoff)_time"
else
  file_str = "2D_sqr_PEPS_Whip_MPO_L$(L)_Chi$(chi)_ZZ_$(min_chi)_$(cutoff)_finer"
end

# jldopen("example.jld2", "w") do f
#     f["a"] = 1
#     f["b"] = 2
#     f["c"] = c
# end

jldsave("$dir_str/$file_str"*".jld2"; 
    saved_linkdim=LinkDims,
    saved_thetas=THETAS,
    saved_zz=ZZs,
    total_time_sec=elapsed_ns * 1e-9
    # saved_x_x=X_Xs,
    # saved_z=Zs
)