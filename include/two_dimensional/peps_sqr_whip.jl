using JLD2, Printf

include("mps_funcs.jl")
include("peps_funcs.jl")

const ROOT_PATH = "$(homedir())/Documents/QC_whip/code-2d/"

# System parameters
L = parse(Int, ARGS[1])
chi = parse(Int, ARGS[2])
cutoff = 1e-8
N = L*L

# initial state |++...+>
ket_es = twodim_sqr_peps(L, "+"^N);
sites = get_siteinds(ket_es)


THETAS = Array{Float64}([pi/4 * i/50 for i in 000:1:200])
MaxDims = Array{Int64}([])
ZZs = Array{Float64}([])

println("Start iteration...")
# location of the observable
i0, i1 = (L-1)*L+1, (L-1)*L+2
for (i,theta) in enumerate(THETAS)
  psi = deepcopy(ket_es)
  # run the whip circuit
  gates = make_sqr_whip_gates_obc(L, theta)
  zytensors = make_tensormap_of_ZY(gates, sites)
  make_contraction_peps!(gates, zytensors, psi, chi, cutoff)
  # contract to obtain the normalization factor
  val3 = contract_twopeps_naive(psi, psi, chi, cutoff, plev=0)
  # observable
  szpsi = deepcopy(psi)
  szpsi[i0] *= op("Sz", sites[i0])
  noprime!(szpsi[i0], tags="Site")
  szpsi[i1] *= op("Sz", sites[i1])
  noprime!(szpsi[i1], tags="Site")
  szsz_val, linkdims = contract_twopeps_naive(szpsi, psi, chi, cutoff, plev=1)
#   i%10==0 ? println("theta: (round($theta/pi;digits=3)), <Sz_i Sz_j>: $(round(szsz_val/val3;digits=4)), linkdim: $(maximum(linkdims))") : nothing
  if i%10 == 0
    @printf "theta = %1.3f pi, <Sz_i Sz_j> = %1.4f, linkdim = %d \n" (theta/pi) (szsz_val/val3) maximum(linkdims)
  end
  # store data
  push!(MaxDims, maximum(linkdims))
  push!(ZZs, real(szsz_val/val3))
end

dir_str = "$(ROOT_PATH)/data/"
if !isdir("$dir_str")
    mkdir("$dir_str")
end
file_str = "2D_sqr_PEPS_Whip_L$(L)_Chi$(chi)_ZZ"

# jldopen("example.jld2", "w") do f
#     f["a"] = 1
#     f["b"] = 2
#     f["c"] = c
# end

jldsave("$dir_str/$file_str"*".jld2"; 
    saved_linkdim=MaxDims,
    saved_thetas=THETAS,
    saved_zz=ZZs
)