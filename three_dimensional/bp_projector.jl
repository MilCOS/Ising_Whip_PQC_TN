if !isdefined(Main, :DoubleLayerTN)
    include("bp_contraction.jl")
end
if !isdefined(Main, :bp_norm)
    include("bp_direct.jl")
end

function bp_projector_probability(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    projectors::Dict{Int64,Symbol};
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
    pair_normalize::Bool=false,
)
    ops = Dict(site => projector_operator_matrix(state) for (site, state) in projectors)
    return bp_network_value(
        psi,
        cubic;
        operators=ops,
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
end

function bp_zz_projector(
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
    pp = bp_projector_probability(
        psi,
        cubic,
        Dict(i => :up, j => :up);
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
    pm = bp_projector_probability(
        psi,
        cubic,
        Dict(i => :up, j => :down);
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
    mp = bp_projector_probability(
        psi,
        cubic,
        Dict(i => :down, j => :up);
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
    mm = bp_projector_probability(
        psi,
        cubic,
        Dict(i => :down, j => :down);
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
    zz = (pp.value - pm.value - mp.value + mm.value) / denom.value
    return zz, (upup=pp, updown=pm, downup=mp, downdown=mm), denom
end

function bp_sink_zz_projector(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
    pair_normalize::Bool=false,
)
    i, j = last_sink_bond(cubic)
    return bp_zz_projector(psi, cubic, i, j; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=pair_normalize)
end
