using ITensors, ITensorMPS, LinearAlgebra

include("gates.jl")

"""
Functions for initializing the whip circuit
  Normal definition of square lattice
"""
function coor2idx(coor::Tuple{Int64, Int64}, l::Int64)
  x,y = coor
  idx = (x-1)*l + y
  return idx
end
function idx2coor(idx::Int64, l::Int64)
  y = (idx%l == 0) ? l : idx%l
  if idx<=l
    return 1, y
  else
    return Int((idx-y)/l)+1, y
  end
end
function sqr_whip_sfzy_obc(l::Int64)
  x0, y0 = l, l
  coupling_map = Array([])
  for x in 1:x0
    for y in 1:y0
      coor1 = (x, y)
      coor2 = (x+1, y)
      coor3 = (x, y+1)
      if (x<x0) && (y<y0)
        y==1 ? push!(coupling_map, (coor1, coor2, +2.0)) : push!(coupling_map, (coor1, coor2, +1.0)) # right
        #                                         ^^^^^
        x==1 ? push!(coupling_map, (coor1, coor3, +2.0)) : push!(coupling_map, (coor1, coor3, +1.0)) # up
        #                                         ^^^^^
      elseif (x<x0) && (y==y0)
        push!(coupling_map, (coor1, coor2, +1.0)) # right
      elseif (x==x0) && (y<y0)
        push!(coupling_map, (coor1, coor3, +1.0)) # up
      else
        # println(x," ",y)
        nothing
      end
    end
  end
  # generate gate_map
  gate_map = Array([])
  for obj in coupling_map
    coor_out, coor_in, angle = obj
    idx_out, idx_in = coor2idx(coor_out,l),coor2idx(coor_in,l)
    push!(gate_map, (idx_out, idx_in, angle))
  end
  return gate_map, coupling_map
end

function make_sqr_whip_gates_obc(l::Int64, theta::Float64)
    g_map, c_map = sqr_whip_sfzy_obc(l)
    indices = []
    paras = zeros(Float64, length(g_map))
    for (i,obj) in enumerate(g_map)
        idx_out, idx_in, angle = obj
        push!(indices, (idx_out, idx_in))
        paras[i] = theta * angle
    end
    GateNet(length(paras), indices, paras)
end

"""
Initialize the ZY gates
"""
function make_tensormap_of_ZY(gates::GateNet, sites::Vector{Index{Int64}})
    t_mat = zeros(Float64,4,4)
    Tm = Array{ITensor}(undef, gates.Ngates)
    for (idx, coor) in enumerate(gates.indices)
        i, j = coor[1], coor[2]
        s_i, s_j = sites[i], sites[j]
        # Construct a two-qubit gate as a local tensor
        _tensorzy!(t_mat, gates.paras[idx])
        comb_1 = combiner(prime(s_i), prime(s_j))
        comb_2 = combiner(s_i, s_j)
        Tc = ITensor(t_mat, combinedind(comb_1), combinedind(comb_2))
        locT = Tc * dag(comb_1) * dag(comb_2)
        # locT = ITensor(t_mat, prime(s_i), prime(s_j), s_i, s_j)
        Tm[idx] = deepcopy(locT)
    end
    Tm
end


function zy_gate_mpo(θ,l,s_i,s_j,ortho="R")
    l = Index(2, tags="Link,l=$(l)")
    T1 = ITensor(zeros(Float64,8), s_i, prime(s_i), l)
    T2 = ITensor(zeros(Float64,8), s_j, prime(s_j), l)
    if ortho=="R"
        T1[1,1,1] = 1.0
        T1[2,2,1] = 1.0
        T1[1,1,2] = 1.0
        T1[2,2,2] = -1.0
        T2[1,1,1] = cos(θ)
        T2[2,2,1] = cos(θ)
        T2[1,2,2] = sin(θ)
        T2[2,1,2] = -sin(θ)
    elseif ortho=="L"
        T1[1,1,1] = cos(θ)
        T1[2,2,1] = cos(θ)
        T1[1,1,2] = sin(θ)
        T1[2,2,2] = -sin(θ)
        T2[1,1,1] = 1
        T2[2,2,1] = 1
        T2[1,2,2] = 1
        T2[2,1,2] = -1
    elseif ortho=="LR"
        T1[1,1,1] = cos(θ)^0.5
        T1[2,2,1] = cos(θ)^0.5
        T1[1,1,2] = sin(θ)^0.5
        T1[2,2,2] = -sin(θ)^0.5
        T2[1,1,1] = cos(θ)^0.5
        T2[2,2,1] = cos(θ)^0.5
        T2[1,2,2] = sin(θ)^0.5
        T2[2,1,2] = -sin(θ)^0.5
    end
    MPO([T1, T2])
end

function make_mpomap_of_ZY(gates::GateNet, sites::Vector{Index{Int64}}, decom::String="R")
    Tm = Array{MPO}(undef, gates.Ngates)
    for (idx, coor) in enumerate(gates.indices)
        i, j = coor[1], coor[2]
        s_i, s_j = sites[i], sites[j]
        # Construct a two-qubit gate as a MPO
        locT = zy_gate_mpo(gates.paras[idx],idx,s_i,s_j,decom)
        Tm[idx] = deepcopy(locT)
    end
    Tm
end


"""
Generate PEPS
"""
function twodim_sqr_peps(l::Int64,st::String,sites::Vector{T}=[]) where T
  g_map, c_map = sqr_whip_sfzy_obc(l)
  links = [Index(1,"Link,l=$(i)") for i in eachindex(g_map)]
  nbits = l*l
  # bond/link
  loc_links = [[] for i in 1:nbits]
  for (i,obj) in enumerate(g_map)
    q1, q2, _ = obj
    push!(loc_links[q1], links[i])
    push!(loc_links[q2], links[i])
  end
#   println("--bond initialized--")
  # sites
  if sites == []
    loc_sites = siteinds("Qubit", nbits)
  else
    loc_sites = sites
  end
  vals_dict = Dict(["0"=>[1.0,0.0],
                   "1"=>[0.0,1.0],
                   "+"=>[1.0/sqrt(2),1.0/sqrt(2)],
                   "-"=>[1.0/sqrt(2),-1.0/sqrt(2)]])
  loc_states = split(st, "") # 0, 1, +, -
  loc_vals = [vals_dict[s] for s in loc_states]
#   println("--sites initialized--")
  # Tensors
  tensors = ITensor[]
  for i in 1:nbits
    link_idx = loc_links[i]
    site_idx = loc_sites[i]
    vals = loc_vals[i]
    T1 = ITensor(vals, site_idx, link_idx...)
    push!(tensors, T1)
  end
  tensors
end

"""
Apply a two-qubit gate to the PEPS at site i and j
"""
function make_contraction_peps!(gates::GateNet, Tm::Array{ITensor}, peps::Vector{ITensor}, chi::Int64, cutoff::Float64=1E-8)
  # Apply sitewise tensor to the PEPS
  for idx in 1:gates.Ngates
    G2 = Tm[idx] # tensor of a two-qubit gate
    i, j = gates.indices[idx] # site indices
    @assert i<j
    peps_local_contraction!(peps, G2, i, j, chi, cutoff)
  end
  # for idx in 1:gates.Ngates
  #   i, j = gates.indices[idx] # site indices
  #   normalize_local_tensor!(peps, [i,j])
  # end
end

function peps_local_contraction!(psi::Vector{ITensor}, locT2::ITensor, i::Int64, j::Int64, chi::Int64, cutoff::Float64=1E-8)
  newT2 = locT2 * (psi[i]*psi[j]) # contract
  noprime!(newT2)
  indsij = uniqueinds(psi[i],psi[j])
  U,Q = factorize(newT2, indsij, ortho="none", which_decomp="svd", maxdim=chi, cutoff=cutoff)
  psi[i] = U
  psi[j] = Q
end

function make_mpo_contraction_peps!(gates::GateNet, Tm::Array{MPO}, peps::Vector{ITensor})
  # Apply sitewise tensor to the PEPS
  for idx in 1:gates.Ngates
    G2 = Tm[idx] # tensor of a two-qubit gate
    i, j = gates.indices[idx] # site indices
    @assert i<j
    peps_local_mpo_contraction!(peps, G2)
  end
  # for idx in 1:gates.Ngates
  #   i, j = gates.indices[idx] # site indices
  #   normalize_local_tensor!(peps, [i,j])
  # end
end

function peps_local_mpo_contraction!(psi::Vector{ITensor}, mpo2q::MPO)
  sites = get_siteinds(psi)
  u, q = mpo2q[1], mpo2q[2]
  s_i = filter(ind -> (plev(ind)==0 && hastags(ind, "Site")), inds(u))[1]
  qi = findfirst(==(s_i), sites)
  s_j = filter(ind -> (plev(ind)==0 && hastags(ind, "Site")), inds(q))[1]
  qj = findfirst(==(s_j), sites)
  newTi = psi[qi] * u
  noprime!(newTi)
  newTj = psi[qj] * q
  noprime!(newTj)
  linkinds = commoninds(newTi,newTj)
  comb_now = combiner(linkinds; tags=tags(linkinds[1]))
  U = newTi * comb_now
  Q = newTj * comb_now
  psi[qi] = U
  psi[qj] = Q
end


function normalize_local_tensor!(peps::Vector{ITensor}, indices)
  for i in indices
    peps[i] = peps[i] ./ sqrt((peps[i]*peps[i])[1])
  end
end


"""
 o--o--o    o--o
 |  |  | => ‖  | => ...
 o--o--o    o--o
"""
function contract_twopeps_naive(peps1::Vector{ITensor}, peps2::Vector{ITensor}, chi::Int64, cutoff::Float64=1E-8; plev::Int64=0)
  _peps2 = prime.(dag.(peps2)) # Index of peps2 is primed
  noprime!.(_peps2, tags="Site") # unprime site indices
  nbits = length(peps1)
  l = Int(sqrt(nbits))
  pepsc = peps1 .* _peps2 # because of small linkdim
  merge_overlap_links!(pepsc) # merge o==o into o--o
#   println(inds(pepsc[1]))
  pepsc = reshape(pepsc, (l,l)) # reshape to 2d array
  # loop over local tensors
  l_linkdims = Int64[]
  l_col = pepsc[:,1] # the first col
  a, b = 2, l
  for x=a:+1:b
    l_col = l_col .* pepsc[:,x]
    if plev >= 1
      plev >= 2 ? println("Link dimension after contracting column $x: $(get_maxlinkdim(l_col))") : nothing
      push!(l_linkdims, get_maxlinkdim(l_col))
    end
    if x<b
      l_col = reduce_linkdim(l_col, chi, cutoff)
    end
  end
  val = ITensor(1.0)
  for y=1:l
    val = val * l_col[y]
  end
  if plev >= 1
    return val[1], l_linkdims
  else
    return val[1]
  end
end

function reduce_linkdim(l_col::Vector{ITensor}, chi::Int64, cutoff::Float64=1E-8)
  nrow = length(l_col)
  for i in 1:nrow-1
    T2 = l_col[i]*l_col[i+1]
    indsij = uniqueinds(l_col[i],l_col[i+1])
    U,S,V = svd(T2, indsij, cutoff=cutoff, maxdim=chi)
    Q = S*V
    linkid = commonind(l_col[i],l_col[i+1]) # only one common index
    bond = commonind(U,Q) # only one common index
    newbond = replacetags(bond, "Link,u" => tags(linkid))
    replaceind!(U, bond, newbond)
    replaceind!(Q, bond, newbond)
    l_col[i] = U
    l_col[i+1] = Q
  end
  l_col
end

function contract_twopeps_bdr_mps(peps1::Vector{ITensor}, peps2::Vector{ITensor}, chi::Int64, cutoff::Float64=1E-8; plev::Int64=0)
  _peps2 = prime.(dag.(peps2)) # Index of peps2 is primed
  noprime!.(_peps2, tags="Site") # unprime site indices
  nbits = length(peps1)
  l = Int(sqrt(nbits))
  pepsc = peps1 .* _peps2 # because of small linkdim
  merge_overlap_links!(pepsc) # merge o==o into o--o
  pepsc = reshape(pepsc, (l,l)) # reshape to 2d array
  replace_link_with_siteinds!(pepsc) # and replace some links with tags of sites
  # loop over local tensors
  l_linkdims = Int64[]
  l_col = MPS(pepsc[:,1]) # the first col, boundary MPS
  # orthogonalize!(l_col,1)
  # normalize!(l_col)
  r_col = MPS(pepsc[:,end])

  a, b = 2, l-1
  l_coeff = Float64[]
  for x=a:+1:b
    # view this process as MPS-MPO contraction. Use the 'apply' function in ITensor. This function employs the Density-matrix method.
    l_col = apply(MPO(pepsc[:,x]), l_col; cutoff=cutoff, maxdim=chi, mindim=min_chi, alg="densitymatrix")
    push!(l_coeff, (l_col[1]*l_col[1])[1])
    # l_col = l_col ./ l_coeff[end] # divided by normalization factor
    if plev >= 1
      push!(l_linkdims, maxlinkdim(l_col))
      if plev >= 2
        println(x," ",inner(l_col, l_col), " ", inner(r_col, r_col))
        println("Link dimension after contracting column $x: $(maxlinkdim(l_col)) $(maxlinkdim(r_col))")
      end
    end
  end
  val = inner(l_col, r_col)
  if plev >= 1
    return val, l_coeff, l_linkdims
  else
    return val
  end
end

function replace_link_with_siteinds!(pepsc::Matrix{ITensor})
  l = size(pepsc)[1]
  for x=1:l-1
    for y=1:l
      A, B = pepsc[y,x], pepsc[y,x+1]
      bond = commonind(A, B)
      newbond = replacetags(bond, "Link" => "Site")
      replaceind!(A, bond, newbond)
      replaceind!(B, bond, newbond)
      pepsc[y,x] = A
      pepsc[y,x+1] = B
    end
  end
end

function get_statevector_from_peps(nbits::Int, peps::Vector{ITensor}, chi::Int64, cutoff::Float64=1E-8)
    _sites = get_siteinds(peps)
    l = Int(sqrt(nbits))
    # statevec = zeros(typeof(peps[1][1]), 2^nbits)
    statevec = zeros(ComplexF64, 2^nbits)
    for i in eachindex(statevec)
        bitstring = string(i-1, base = 2, pad = nbits) # index to basis state
        # j = parse(Int, join(new_bit), base=2) + 1# basis state to index
        # construct peps based on bitstring
        _peps = twodim_sqr_peps(l, bitstring, _sites)
        # calculate the inner product
        val = contract_twopeps_naive(peps, _peps, chi, cutoff)
        statevec[i] = val
    end
    statevec
end

"""
Find the nonzero bitstring
"""
function check_ground_state_configuration(peps::Vector{ITensor}, chi::Int64, cutoff::Float64=1E-8)
    nbits = length(peps)
    if nbits >= 20
        return print("System size is too large")
    end
    psivec = get_statevector_from_peps(nbits, peps, chi, cutoff)
    nz_loc = []
    for i in eachindex(psivec)
        if isapprox(psivec[i], 0, atol=1E-14)
            continue
        else
            push!(nz_loc, i)
        end
    end
    get_bit.(nz_loc.-1, nbits)
end


"""
make local tensors o==o into o--o
"""
function merge_overlap_links!(peps::Vector{ITensor})
  l = Int(sqrt(length(peps)))
  g_map, _ = sqr_whip_sfzy_obc(l)
  for (i,j,_) in g_map
    linkid = commoninds(peps[i], peps[j])
    C = combiner(linkid...; tags=tags(linkid[1]))
    peps[i] = peps[i] * C
    peps[j] = peps[j] * C
  end
end

"""
Find site indices, maximum link dimension ...
"""
function get_siteinds(peps::Vector{ITensor})
  nbits = length(peps)
  sites = Index{Int64}[]
  for i in 1:nbits
    locinds = inds(peps[i])
    l = collect(hastags.(locinds, "Site"))
    sites = [sites; collect(locinds[l])]
  end
  sites
end

function get_maxlinkdim(l_col::Vector{ITensor})
  nrow = length(l_col)
  maxdim = 0
  for i in 1:nrow-1
    locinds = commoninds(l_col[i], l_col[i+1])
    l = collect(hastags.(locinds, "Link"))
    linkinds = collect(locinds[l])
    d = *(dim.(linkinds)...) # because there may be multiple links
    if d > maxdim
      maxdim = d
    end
  end
  maxdim
end