using ITensors, ITensorMPS, LinearAlgebra


include("gates.jl")


function coor2idx(coor::Tuple{Int64, Int64},l::Int)
  x,y = coor
  if x % 2 == 1
    idx = (x-1)*l + y
  else
    idx = (x-1)*l + (l-y+1)
  end
  return idx
end
"""
Functions for initializing the whip circuit
    Use a defintion that is consistent with the MPS geometry
"""
function sqr_whip_sfzy_obc(l::Int64)
  function coor2idx(coor::Tuple{Int64, Int64})
    x,y = coor
    if x % 2 == 1
      idx = (x-1)*l + y
    else
      idx = (x-1)*l + (l-y+1)
    end
    return idx
  end
  x0, y0 = l, l
  coupling_map = Array([])
  for x in 1:x0
    for y in 1:y0
      coor1 = (x, y)
      coor2 = (x+1, y)
      coor3 = (x, y+1)
      if (x<x0) && (y<y0)
        y==1 ? push!(coupling_map, (coor1, coor2, +2.0)) : push!(coupling_map, (coor1, coor2, +1.0)) # right
        x==1 ? push!(coupling_map, (coor1, coor3, +2.0)) : push!(coupling_map, (coor1, coor3, +1.0))     # up
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
    idx_out, idx_in = coor2idx(coor_out),coor2idx(coor_in)
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
Functions for tensor contractions
"""
function sitewise_local_contraction!(mps::MPS, locT2::ITensor, i::Int64, j::Int64, chi::Int64, cutoff::Float64=1E-16)
    # if mps = orthogonalize(mps, i), the 'mps' variable will be a local variable
    orthogonalize!(mps, i) # first, gauge the MPS so that either site i or j is the orthogonality center
    newT2 = locT2 * (mps[i]*mps[j]) # contract
    locsites = [siteind(mps, i) siteind(mps, j)]
    if plev(locsites[1]) == 0
        # left mps
        noprime!(newT2)
    else
        # right mps
        locsites0 = setprime.(locsites, 0)
        prime!(newT2, locsites0[1], locsites0[2])
    end
    indsij = uniqueinds(mps[i],mps[j])
    U,S,V = svd(newT2, indsij, cutoff=cutoff, maxdim=chi)
    Q = S*V
    linkid = commonind(mps[i],mps[j])
    bond = commonind(U,Q)
    newbond = replacetags(bond, "Link,u" => tags(linkid))
    replaceind!(U, bond, newbond)
    replaceind!(Q, bond, newbond)
    mps[i] = U
    mps[j] = Q
end
function sitewise_local_contraction!(mps::MPS, locT1::ITensor, i::Int64)
    orthogonalize!(mps, i) # first, gauge the MPS so that site i is the orthogonality center
    newT1 = locT1 * mps[i]
    locsite = siteind(mps, i)
    if plev(locsite) == 0
        # left mps
        noprime!(newT1)
    else
        # right mps
        locsite0 = setprime.(locsite, 0)
        prime!(newT1, locsite0)
    end
    mps[i] = newT1
end

function insert_identity_mpo(sites::Vector{Index{Int64}}, nonlocT2::ITensor, qi::Int, qj::Int, chi::Int64, cutoff::Float64=1E-16)
    N = length(sites)
    tensors = ITensor[]
    u,s,v = svd(nonlocT2, [prime(sites[qi]),sites[qi]]; cutoff=cutoff, maxdim=chi) # the "left indices" provided collectively as a row index, and the remaining "right indices" as a column index
    q = s*v
    link_indice = filter(ind -> hastags(ind, "Link"), inds(q))[1]
    link_dim = dim(link_indice)
    link_u = Index(link_dim, "Link,l=$(qi)")
    link_q = Index(link_dim, "Link,l=$(qj-1)")
    right_link = undef
    left_link = undef
    for i in qi:qj
        if i == qi
            # First site: no left link index
            right_link = link_u
            replaceind!(u, link_indice, right_link)
            T = u
        elseif i == qi+1
            right_link = Index(link_dim, "Link,l=$(i)")
            T = ITensor(left_link, sites[i]', sites[i], right_link)
            for l in 1:link_dim
                T[left_link => l, sites[i]' => 1, sites[i] => 1, right_link => l] = 1.0
                T[left_link => l, sites[i]' => 2, sites[i] => 2, right_link => l] = 1.0
            end
        elseif i == qj-1
            right_link = link_q
            T = ITensor(left_link, sites[i]', sites[i], right_link)
            for l in 1:link_dim
                T[left_link => l, sites[i]' => 1, sites[i] => 1, right_link => l] = 1.0
                T[left_link => l, sites[i]' => 2, sites[i] => 2, right_link => l] = 1.0
            end
        elseif i == qj
            # Last site: no right link index
            replaceind!(q, link_indice, left_link)
            T = q
        else
            # Middle sites: both left and right link indices, insert identity
            right_link = Index(link_dim, "Link,l=$(i)")
            T = ITensor(left_link, sites[i]', sites[i], right_link)
            for l in 1:link_dim
                T[left_link => l, sites[i]' => 1, sites[i] => 1, right_link => l] = 1.0
                T[left_link => l, sites[i]' => 2, sites[i] => 2, right_link => l] = 1.0
            end
        end
        left_link = copy(right_link)
        push!(tensors, T)
    end
    return MPO(tensors)
end
# Method 2: Manual construction with proper bond dimensions
function insert_identity_mpo_and_expand(sites::Vector{Index{Int64}}, nonlocT2::ITensor, qi::Int, qj::Int, chi::Int64, cutoff::Float64=1E-16)
    N = length(sites)
    tensors = ITensor[]
    u,s,v = svd(nonlocT2, [prime(sites[qi]),sites[qi]]; cutoff=1E-32) # the "left indices" provided collectively as a row index, and the remaining "right indices" as a column index
    q = s*v
    link_indice = filter(ind -> hastags(ind, "Link"), inds(q))[1]
    link_dim = dim(link_indice)
    link_u = Index(link_dim, "Link,l=$(qi)")
    replaceind!(u, link_indice, link_u)
    link_q = Index(link_dim, "Link,l=$(qj-1)")
    replaceind!(q, link_indice, link_q)
    link_inds = []
    for i in 1:N
        if (i >= qi) && (i+1 <= qj)
            if i == qi
              push!(link_inds, link_u) # link start from u
            elseif i+1 == qj
              push!(link_inds, link_q) # link end at q
            else
              push!(link_inds, Index(link_dim, "Link,l=$(i)")) # link within nonlocT2
            end
        else
            push!(link_inds, Index(1, "Link,l=$(i)"))
        end
    end
    right_link = undef
    left_link = undef
    if qi != 1
        # add left_link if not on the most left
        left_link = link_inds[qi-1]
        right_link = link_inds[qi]
        T = ITensor(left_link, sites[qi]', sites[qi], right_link)
        for idx in Base.product(1:link_dim,1:2,1:2)
            inds_u = (sites[qi]' => idx[2], sites[qi] => idx[3], right_link => idx[1])
            T[left_link => 1, inds_u...] = u[inds_u...]
        end
        u = copy(T)
    end
    if qj != N
        # add right_link if not on the most right
        left_link = link_inds[qj-1]
        right_link = link_inds[qj]
        T = ITensor(left_link, sites[qj]', sites[qj], right_link)
        for idx in Base.product(1:link_dim,1:2,1:2)
            inds_q = (left_link => idx[1], sites[qj]' => idx[2], sites[qj] => idx[3])
            T[inds_q..., right_link => 1] = q[inds_q...]
        end
        q = copy(T)
    end
    for i in 1:N
        if i == 1
            # First site: no left link index
            right_link = link_inds[i]
            if i != qi
              T = ITensor(sites[i]', sites[i], right_link)
              T[sites[i]' => 1, sites[i] => 1, right_link => 1] = 1.0
              T[sites[i]' => 2, sites[i] => 2, right_link => 1] = 1.0
            else
              T = copy(u)
            end
        elseif i == N
            # Last site: no right link index
            left_link = link_inds[i-1]
            if i != qj
              T = ITensor(left_link, sites[i]', sites[i])
              T[left_link => 1, sites[i]' => 1, sites[i] => 1] = 1.0
              T[left_link => 1, sites[i]' => 2, sites[i] => 2] = 1.0
            else
              T = copy(q)
            end
        else
            # Middle sites: both left and right link indices
            left_link = link_inds[i-1]
            right_link = link_inds[i]
            if i == qi
                T = copy(u)
            elseif i == qi+1
                T = ITensor(left_link, sites[i]', sites[i], right_link)
                for l in 1:dim(right_link)
                    T[left_link => l, sites[i]' => 1, sites[i] => 1, right_link => l] = 1.0
                    T[left_link => l, sites[i]' => 2, sites[i] => 2, right_link => l] = 1.0
                end
            elseif i == qj-1
                T = ITensor(left_link, sites[i]', sites[i], right_link)
                for l in 1:dim(right_link)
                    T[left_link => l, sites[i]' => 1, sites[i] => 1, right_link => l] = 1.0
                    T[left_link => l, sites[i]' => 2, sites[i] => 2, right_link => l] = 1.0
                end
            elseif i == qj
                T = copy(q)
            else
                # insert identity
                T = ITensor(left_link, sites[i]', sites[i], right_link)
                for l in 1:dim(right_link)
                    T[left_link => l, sites[i]' => 1, sites[i] => 1, right_link => l] = 1.0
                    T[left_link => l, sites[i]' => 2, sites[i] => 2, right_link => l] = 1.0
                end
            end
        end
        push!(tensors, T)
    end
    return MPO(tensors)
end
function sitewise_nonlocal_contraction!(mps::MPS, nonlocT2::ITensor, qi::Int, qj::Int, chi::Int64, cutoff::Float64=1E-16)
    # construct MPO and fill the intermediate sites with identities
    # [u] -- link,u -- [q]
    sites = siteinds(mps)
    extend_mpo = insert_identity_mpo(sites, nonlocT2, qi, qj, chi, cutoff)
    for (i,loc_i) in enumerate(qi:qj)
        T = replaceprime(extend_mpo[i]*mps[loc_i], 1=>0)
        mps[loc_i] = T
    end
  # contract and truncate
  for loc_i in qi:qj-1
    # orthogonalize!(mps, loc_i) # first, gauge the MPS so that either site i or j is the orthogonality center
    T1 = mps[loc_i]
    T2 = mps[loc_i+1]
    indsij = uniqueinds(T1, T2)
    U,S,V = svd(T1*T2, indsij, cutoff=cutoff, maxdim=chi)
    Q = S*V
    linkid = commonind(T1, T2)
    bond = commonind(U,Q)
    newbond = replacetags(bond, "Link,u" => tags(linkid))
    replaceind!(U, bond, newbond)
    replaceind!(Q, bond, newbond)
    mps[loc_i] = U
    mps[loc_i+1] = Q
  end
    # Method 2
    # extend_mpo = insert_identity_mpo_and_expand(sites, nonlocT2, qi, qj, chi, cutoff)
    # mps .= replaceprime(contract(extend_mpo, mps; maxdim=chi, cutoff=cutoff, method="densitymatrix"), 1=>0) # densitymatrix, naive
    # Method 3: build-in function
    # mps .= apply(extend_mpo, mps; maxdim=chi, cutoff=cutoff)
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


"""
Functions for preparing the MPS of |++...+>
"""
function mps_equal_superposition(N::Int64)
    sites = siteinds("S=1/2", N)
    # 000 state
    states = ["Up" for n in 1:N]
    ket1 = MPS(sites, states)
    for i in 1:N
        s_1 = sites[i]
        locT1 = ITensor(h_gate, prime(s_1), s_1)
        sitewise_local_contraction!(ket1, locT1, i)
    end
    return ket1
end

"""
Functions for executing the whip circuit
"""
function make_contraction!(gates::GateNet, Tm::Array{ITensor}, mps::MPS, chi::Int64, cutoff::Float64=1E-16, checklinkdim::Bool=true)
    # Apply sitewise tensor to the MPS
    for idx in 1:gates.Ngates
        G2 = Tm[idx] # tensor of a two-qubit gate
        i, j = gates.indices[idx] # site indices
        if abs(i-j)!=1
            sitewise_nonlocal_contraction!(mps, G2, i, j, chi, cutoff)
        else
            sitewise_local_contraction!(mps, G2, i, j, chi, cutoff)
        end
        if checklinkdim && (idx==gates.Ngates)
            # println(mps)
            println("link dimensions: ", maxlinkdim(mps))
            # @assert 1==0
        end
    end
end

function make_contraction_buildin!(gates::GateNet, Tm::Array{ITensor}, mps::MPS, chi::Int64, cutoff::Float64=1E-16, checklinkdim::Bool=true)
    # Apply sitewise tensor to the MPS
    for idx in 1:gates.Ngates
        G2 = Tm[idx] # tensor of a two-qubit gate
        mps .= apply(G2, mps; maxdim=chi, cutoff=cutoff)
        if checklinkdim && (idx==gates.Ngates)
            println("link dimensions: ", maxlinkdim(mps))
        end
    end
end

# ==== ====
"""
Functions for projecting back to the computational basis
"""
function contract_twomps(mps1::MPS, mps2::MPS)
    sites = siteinds(mps2) # Index of mps2 is primed
    nbits = length(sites)
    # loop over local tensors
    Tup = ITensor(1.0) # up->bottom
    for i=1:nbits
        Tup = Tup * (mps1[i] * noprime(mps2[i], sites[i]))
    end
    Tup
end

function get_statevector_from_mps(nbits::Int, mps::MPS)
    _sites = siteinds(mps)
    # statevec = zeros(typeof(mps[1][1]), 2^nbits)
    statevec = zeros(ComplexF64, 2^nbits)
    for i in eachindex(statevec)
        bit = string(i-1, base = 2, pad = nbits) # index to basis state
        # j = parse(Int, join(new_bit), base=2) + 1# basis state to index
        # construct mps based on bitstring
        bitstring = split(bit, "")
        imps = MPS(_sites, bitstring)
        # calculate the inner product
        # val = inner(imps, mps)
        val = contract_twomps(mps, imps')[1]
        statevec[i] = val
    end
    statevec
end

"""
Find the nonzero bitstring
"""
function check_ground_state_configuration(mps::MPS)
    println(inner(mps,mps))
    nbits = length(mps)
    if nbits >= 20
        return print("System size is too large")
    end
    psivec = get_statevector_from_mps(nbits, mps)
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
Direct sum
"""
function mps_subtraction(psi1::MPS, psi2::MPS)
    sites = siteinds(psi1)
    nsites = size(psi1)[1]
    psi3 = MPS(sites)
    for i in 1:nsites
        T1 = psi1[i]
        T2 = psi2[i]
        linds1 = uniqueinds(T1,T2)
        sort_inds!(linds1)
        tags = [linds1[i].tags for i in eachindex(linds1)]
        linds2 = uniqueinds(T2,T1)
        sort_inds!(linds2)
        T3, s3 = directsum(T1 => linds1, T2 => linds2; tags=tags)
        # linds3 = uniqueinds(T3,T1)
        # replaceinds!(T3, linds3, linds1)
        psi3[i] = T3
    end
    standardize_indices!(psi3)
    orthogonalize!(psi3, 1)
    return psi3
end

function sort_inds!(linds::Vector{Index{Int}})
    ls = []
    for lind in linds
        s = String( lind.tags[2] ) # tag[1]: Link, tag[2]: l=1
        idx = last(findfirst("l=", s))
        l = parse(Int, s[idx+1:end])
        push!(ls, l)
    end
    p = sortperm(ls)
    @views linds[:] = linds[p]
end

function standardize_indices!(mps::MPS)
    canonical = nothing
    canontags = nothing
    sites = siteinds(mps)
    for i in eachindex(mps)
        # Loop over all indices in the current tensor
        linds = uniqueinds(mps[i], sites[i])
        alltags = [linds[i].tags for i in eachindex(linds)]
        if i == 1
            canonical = [ind for ind in linds]
            canontags = [canonical[i].tags for i in eachindex(canonical)]
        else
            # Find the same links, update the index id
            jk = indexin(canontags, alltags)
            for j in eachindex(jk)
                # Replace this index with the canonical one
                replaceind!(mps[i], linds[jk[j]], canonical[j])
            end
            # Find different links, update canonical and tag
            canonical = []
            for ind in linds
                if ind.tags ∉ canontags
                    push!(canonical, ind)
                end
            end
            canontags = [canonical[i].tags for i in eachindex(canonical)]
        end
    end
end


"""
Entanglement entropy
https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html#Computing-the-Entanglement-Entropy-of-an-MPS
"""
function entanglement_entropy_svd(mps::MPS, cut_bond)
    b = cut_bond #Int(N/2)
    mps_copy = orthogonalize(mps, b)
    # The reduced density matrix eigenvalues are the squares of the
    # Schmidt coefficients, which are the singular values
    U,S,V = svd(mps_copy[b], (linkinds(mps_copy, b-1)..., siteinds(mps_copy, b)...))
    # Extract Schmidt coefficients (singular values)
    S_vN = 0.0
    for n=1:dim(S, 1)
      p = S[n,n]^2   # Probability = |coefficient|^2
    #   if p > 1E-16 # Avoid log(0)
          S_vN -= p * log2(p)
    #   end
    end
    S_vN, diag(S)
end