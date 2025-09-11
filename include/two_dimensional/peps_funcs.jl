"""
Generate PEPS
"""
function twodim_sqr_peps(l::Int64,st::String,sites::Vector{T}=[]) where T
  g_map, _ = sqr_whip_sfzy_obc(l)
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
    peps_local_contraction!(peps, G2, i, j, chi, cutoff)
  end
end

function peps_local_contraction!(psi::Vector{ITensor}, locT2::ITensor, i::Int64, j::Int64, chi::Int64, cutoff::Float64=1E-8)
  newT2 = locT2 * (psi[i]*psi[j]) # contract
  noprime!(newT2)
  indsij = uniqueinds(psi[i],psi[j])
  U,S,V = svd(newT2, indsij, cutoff=cutoff, maxdim=chi)
  Q = S*V
  linkid = commonind(psi[i],psi[j]) # only one common index
  bond = commonind(U,Q) # only one common index
  newbond = replacetags(bond, "Link,u" => tags(linkid))
  replaceind!(U, bond, newbond)
  replaceind!(Q, bond, newbond)
  psi[i] = U
  psi[j] = Q
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
    if x%2==0 # contract next col in reverse order
      l_col = l_col .* reverse(pepsc[:,x])
    else # contract next col
      l_col = l_col .* pepsc[:,x]
    end
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