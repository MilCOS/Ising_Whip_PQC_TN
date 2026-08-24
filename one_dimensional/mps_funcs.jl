using ITensors, ITensorMPS, LinearAlgebra


"""
Quantum Gates
"""
const h_gate = [1 1; 1 -1] ./ sqrt(2)

RIX(θ::Float64) = Array{ComplexF64}([cos(θ/2) -1im*sin(θ/2) 0 0; 
                                    -1im*sin(θ/2) cos(θ/2) 0 0;
                                    0 0 cos(θ/2) -1im*sin(θ/2);
                                    0 0 -1im*sin(θ/2) cos(θ/2)
                                    ])
const rix_p = RIX(pi/2)
ZZPhase(θ::Float64) = Array{ComplexF64}([exp(-1im/2*θ) 0 0 0;
                                    0 exp(+1im/2*θ) 0 0;
                                    0 0 exp(+1im/2*θ) 0;
                                    0 0 0 exp(-1im/2*θ)
                                    ])
get_int(s) = parse(Int, s, base=2)
get_bit(idx, n) = string(idx, base=2, pad=n)

mutable struct GateNet
    Ngates::Int64 # number of gates each layer
    indices::Array{Tuple{Int64,Int64}} # (site_i,site_j)_idx 
    paras::Array{Float64,1} # idx
end
function _tensorzy!(tmat::Matrix{Float64}, a::Real)
    # exp(-i a/2 ZY) = RIX(-pi/2) ZZPhase(a) RIX(pi/2)
    if all(a .== 0)
        umat = Matrix{Float64}(I(4))
    else
        umat = real(rix_p' * ZZPhase(a) * rix_p)
    end
    gate2mat!(tmat, umat)
end
function gate2mat!(tmat::Matrix{Float64},umat::Matrix{Float64})
    """  (i j) -> (si' sj') (si sj)
    Note the unitary gate is defined as
    |00>' = c0000|00> + c0001|01> + c0010|10> + c0011|11>
    |01>' = c0100|00> + c0101|01> + c0110|10> + c0111|11>
    |10>' = c1000|00> + c1001|01> + c1010|10> + c1011|11>
    |11>' = c1100|00> + c1101|01> + c1110|10> + c1111|11>
    The elements stored in Tensor look like 0000 1000 0100 1100 ...
    If we directly reshape the unitary into a matrix in row-first order, the matrix elements 
    would not match the basis: (00,01,10,11)
    """
    basis = ["00","01","10","11"]
    for (i,bi) in enumerate(basis) # si'sj'
        for (j,bj) in enumerate(basis) # si sj
            s = reverse(bi * bj)
            idx = get_int(s) + 1 # because 000 = 1
            tmat[idx] = umat[i,j]
        end
    end
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

# Method 2: Manual construction with proper bond dimensions
function insert_identity_mpo_manual(sites::Vector{Index{Int64}}, nonlocT2::ITensor, qi::Int, qj::Int, chi::Int64, cutoff::Float64=1E-16)
    N = length(sites)
    tensors = ITensor[]
    u,s,v = svd(nonlocT2, [prime(sites[qi]),sites[qi]]; cutoff=cutoff, maxdim=chi) # the "left indices" provided collectively as a row index, and the remaining "right indices" as a column index
    q = s*v
    link_indice = filter(ind -> hastags(ind, "Link"), inds(q))[1]
    link_dim = dim(link_indice)
    right_link = undef
    left_link = undef
    for i in qi:qj
        if i == qi
            right_link = Index(link_dim, "Link,l=$(i)")
            # First site: no left link index
            replaceind!(u, link_indice, right_link)
            T = u
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
function sitewise_nonlocal_contraction!(mps::MPS, nonlocT2::ITensor, qi::Int, qj::Int, chi::Int64, cutoff::Float64=1E-16)
    # construct MPO and fill the intermediate sites with identities
    # [u] -- link,u -- [q]
    sites = siteinds(mps)
    extend_mpo = insert_identity_mpo_manual(sites, nonlocT2, qi, qj, chi, cutoff)
    for i in eachindex(sites)
        T = extend_mpo[i] * mps[i]
        mps[i] = replaceprime(T, 1=>0)
    end
    # mps .= apply(extend_mpo, mps)
end

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
function mps_equal_superposition(N::Int64, isopen=true, use_nl_op=false)
    sites = siteinds("S=1/2", N)
    # 000 state
    states = ["Up" for n in 1:N]
    ket0 = MPS(sites, states)
    if !isopen
        link1N = ITensor(1, Index(1, "Link,l=$(N)")) # Add periodic link
        ket0[1] *= link1N
        ket0[N] *= link1N
    end
    ket1 = deepcopy(ket0)
    for i in 1:N
        s_1 = sites[i]
        locT1 = ITensor(h_gate, prime(s_1), s_1)
        sitewise_local_contraction!(ket1, locT1, i)
    end
    return ket1
end

"""
Functions for initializing the whip circuit
"""
function make_1d_whip_gates(nbits::Int64, theta::Float64, use_nl_op=true)
    if nbits == 2
        paras = Array([theta*2])
        indices = [(1,nbits)]
    else
        if use_nl_op
            indices = [(1,nbits)]
            paras = zeros(Float64, nbits)
            paras[1] = theta
            paras[2:end-1] .= theta * 2
            paras[end] = theta
        else
            indices = []
            paras = zeros(Float64, nbits-1)
            paras .= theta * 2
        end
        for i in 1:nbits-1
            push!(indices, (i,i+1))
        end
    end
    @assert length(paras) == length(indices)
    GateNet(length(paras), indices, paras)
end

"""
Functions for executing the whip circuit
"""
function make_contraction!(gates::GateNet, Tm::Array{ITensor}, mps::MPS, chi::Int64, cutoff::Float64=1E-16, isopen::Bool=true)
    # Apply sitewise tensor to the MPS
    for idx in 1:gates.Ngates
        G2 = Tm[idx] # tensor of a two-qubit gate
        i, j = gates.indices[idx] # site indices
        if abs(i-j)!=1 && isopen
            sitewise_nonlocal_contraction!(mps, G2, i, j, chi, cutoff)
        else
            sitewise_local_contraction!(mps, G2, i, j, chi, cutoff)
        end
    end
end

function make_contraction_buildin!(gates::GateNet, Tm::Array{ITensor}, mps::MPS, chi::Int64, cutoff::Float64=1E-16)
    # Apply sitewise tensor to the MPS
    for idx in 1:gates.Ngates
        G2 = Tm[idx] # tensor of a two-qubit gate
        mps .= apply(G2, mps; maxdim=chi, cutoff=cutoff)
    end
end

# ==== ====
"""
Functions for projecting back to the computational basis
"""
function get_statevector_from_mps(nbits::Int, mps::MPS, isopen::Bool=true)
    _sites = siteinds(mps)
    # statevec = zeros(typeof(mps[1][1]), 2^nbits)
    statevec = zeros(ComplexF64, 2^nbits)
    for i in eachindex(statevec)
        bit = string(i-1, base = 2, pad = nbits) # index to basis state
        # j = parse(Int, join(new_bit), base=2) + 1# basis state to index
        # construct mps based on bitstring
        bitstring = split(bit, "")
        imps = MPS(_sites, bitstring)
        if !isopen
            link1N = ITensor(1, Index(1, "Link,l=$(N)")) # Add periodic link
            imps[1] *= link1N
            imps[nbits] *= link1N
        end
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
function check_ground_state_configuration(mps::MPS, isopen::Bool=true)
    println(inner(mps,mps))
    nbits = length(mps)
    if nbits >= 20
        return print("System size is too large")
    end
    psivec = get_statevector_from_mps(nbits, mps, isopen)
    nz_loc = []
    for i in eachindex(psivec)
        if isapprox(psivec[i], 0, atol=1E-12)
            continue
        else
            push!(nz_loc, i)
        end
    end
    get_bit.(nz_loc.-1, nbits)
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