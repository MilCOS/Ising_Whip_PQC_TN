if !isdefined(Main, :DoubleLayerTN)
    include("bp_contraction.jl")
end
if !isdefined(Main, :bp_projector_probability)
    include("bp_projector.jl")
end

struct SimpleLoop
    vertices::Vector{Int64}
    edges::Vector{Int64}
    key::Tuple{Vararg{Int64}}
end

struct LoopBPResult
    bp::BPResult
    loops::Vector{SimpleLoop}
    loop_corrections::Vector{Float64}
    value::Float64
end

function _edge_id(cubic::CubicWhipGraph, u::Int64, v::Int64)
    return _edge_lookup(cubic)[_sorted_edge(u, v)]
end

function _cycle_edge_ids(cubic::CubicWhipGraph, vertices::Vector{Int64})
    n = length(vertices)
    return [
        _edge_id(cubic, vertices[i], vertices[mod1(i + 1, n)])
        for i in 1:n
    ]
end

function _canonical_loop_key(edge_ids::Vector{Int64})
    return Tuple(sort(edge_ids))
end

function graph_short_loops(cubic::CubicWhipGraph, lmax::Int64)
    @assert lmax >= 4
    raw_cycles = simplecycles_limited_length(cubic.graph, lmax)
    loops = Dict{Tuple{Vararg{Int64}},SimpleLoop}()

    for cycle in raw_cycles
        vertices = collect(Int64, cycle)
        length(vertices) >= 4 || continue
        length(vertices) <= lmax || continue

        edge_ids = _cycle_edge_ids(cubic, vertices)
        key = _canonical_loop_key(edge_ids)
        if !haskey(loops, key)
            loops[key] = SimpleLoop(vertices, edge_ids, key)
        end
    end
    return collect(values(loops))
end

function _edge_position(tn::DoubleLayerTN, site::Int64, edge_id::Int64)
    pos = findfirst(==(edge_id), tn.incident_edges[site])
    @assert pos !== nothing
    return pos
end

function _reduced_factor_for_loop_edges(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    site::Int64,
    left_edge_id::Int64,
    right_edge_id::Int64,
)
    factor = tn.factors[site]
    neighbors = tn.incident_neighbors[site]
    left_pos = _edge_position(tn, site, left_edge_id)
    right_pos = _edge_position(tn, site, right_edge_id)
    reduced = zeros(Float64, size(factor, left_pos), size(factor, right_pos))

    for ci in CartesianIndices(factor)
        inds = Tuple(ci)
        weight = factor[ci]
        for pos in eachindex(neighbors)
            if pos == left_pos || pos == right_pos
                continue
            end
            weight *= messages[(neighbors[pos], site)][inds[pos]]
        end
        reduced[inds[left_pos], inds[right_pos]] += weight
    end
    return reduced
end

function _q_matrix(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    left::Int64,
    right::Int64,
    edge_id::Int64,
)
    q = tn.edge_dims[edge_id]
    qmat = Matrix{Float64}(I, q, q)
    qmat .-= messages[(right, left)] * transpose(messages[(left, right)])
    return qmat
end

function simple_loop_correction(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    loop::SimpleLoop,
)
    n = length(loop.vertices)
    @assert n == length(loop.edges)

    transfer = nothing
    for i in 1:n
        site = loop.vertices[i]
        prev_edge = loop.edges[mod1(i - 1, n)]
        next_edge = loop.edges[i]
        next_site = loop.vertices[mod1(i + 1, n)]

        reduced = _reduced_factor_for_loop_edges(tn, messages, site, prev_edge, next_edge)
        qmat = _q_matrix(tn, messages, site, next_site, next_edge)
        local_transfer = reduced * qmat
        transfer = transfer === nothing ? local_transfer : transfer * local_transfer
    end
    return tr(transfer)
end

function loop_contraction_value(
    tn::DoubleLayerTN,
    cubic::CubicWhipGraph;
    lmax::Int64=4,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    bp = run_bp(tn; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=true)
    loops = graph_short_loops(cubic, lmax)
    corrections = [
        simple_loop_correction(tn, bp.messages, loop)
        for loop in loops
    ]
    return LoopBPResult(bp, loops, corrections, bp.value + sum(corrections))
end

function loop_norm(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    lmax::Int64=4,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    tn = double_layer_tn(psi, cubic)
    return loop_contraction_value(tn, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
end

function loop_zz(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    i::Int64,
    j::Int64;
    lmax::Int64=4,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    denom = loop_norm(psi, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
    numer_tn = double_layer_tn(psi, cubic; operators=Dict(i => z_operator_matrix(), j => z_operator_matrix()))
    numer = loop_contraction_value(numer_tn, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
    return numer.value / denom.value, numer, denom
end

function loop_sink_zz(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    lmax::Int64=4,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    i, j = last_sink_bond(cubic)
    return loop_zz(psi, cubic, i, j; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
end

function loop_zz_projector(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    i::Int64,
    j::Int64;
    lmax::Int64=4,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    denom = loop_norm(psi, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
    pp_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:up), j => projector_operator_matrix(:up)))
    pm_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:up), j => projector_operator_matrix(:down)))
    mp_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:down), j => projector_operator_matrix(:up)))
    mm_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:down), j => projector_operator_matrix(:down)))

    pp = loop_contraction_value(pp_tn, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
    pm = loop_contraction_value(pm_tn, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
    mp = loop_contraction_value(mp_tn, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
    mm = loop_contraction_value(mm_tn, cubic; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
    zz = (pp.value - pm.value - mp.value + mm.value) / denom.value
    return zz, (upup=pp, updown=pm, downup=mp, downdown=mm), denom
end

function loop_sink_zz_projector(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    lmax::Int64=4,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    i, j = last_sink_bond(cubic)
    return loop_zz_projector(psi, cubic, i, j; lmax=lmax, maxiter=maxiter, tol=tol, damping=damping)
end
