
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
