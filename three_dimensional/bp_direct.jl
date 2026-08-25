if !isdefined(Main, :DoubleLayerTN)
    include("bp_contraction.jl")
end

function bp_norm(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
    pair_normalize::Bool=false,
)
    return bp_network_value(
        psi,
        cubic;
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
end

function bp_zz(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    i::Int64,
    j::Int64;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
    pair_normalize::Bool=false,
)
    denom = bp_norm(psi, cubic; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=pair_normalize)
    numer = bp_network_value(
        psi,
        cubic;
        operators=Dict(i => z_operator_matrix(), j => z_operator_matrix()),
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
    return numer.value / denom.value, numer, denom
end

function bp_sink_zz(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
    pair_normalize::Bool=false,
)
    i, j = last_sink_bond(cubic)
    return bp_zz(psi, cubic, i, j; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=pair_normalize)
end
