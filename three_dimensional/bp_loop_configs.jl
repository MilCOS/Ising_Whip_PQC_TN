if !isdefined(Main, :DoubleLayerTN)
    include("bp_contraction.jl")
end
if !isdefined(Main, :bp_projector_probability)
    include("bp_projector.jl")
end

struct LoopConfig
    edges::Vector{Int64}
    vertices::Vector{Int64}
    key::Tuple{Vararg{Int64}}
end

struct LoopConfigBPResult
    bp::BPResult
    configs::Vector{LoopConfig}
    relative_corrections::Vector{Float64}
    value::Float64
end

function _config_edge_id(cubic::CubicWhipGraph, u::Int64, v::Int64)
    return _edge_lookup(cubic)[_sorted_edge(u, v)]
end

function _config_cycle_edge_ids(cubic::CubicWhipGraph, vertices::Vector{Int64})
    n = length(vertices)
    return [
        _config_edge_id(cubic, vertices[i], vertices[mod1(i + 1, n)])
        for i in 1:n
    ]
end

_config_key(edge_ids) = Tuple(sort(collect(Int64, edge_ids)))

function _config_vertices(cubic::CubicWhipGraph, edge_ids::Vector{Int64})
    touched = Set{Int64}()
    for edge_id in edge_ids
        u, v = cubic.bonds[edge_id]
        push!(touched, u)
        push!(touched, v)
    end
    return sort!(collect(touched))
end

function _is_nonzero_loop_config(cubic::CubicWhipGraph, edge_ids::Vector{Int64})
    degrees = Dict{Int64,Int64}()
    for edge_id in edge_ids
        u, v = cubic.bonds[edge_id]
        degrees[u] = get(degrees, u, 0) + 1
        degrees[v] = get(degrees, v, 0) + 1
    end
    return all(!=(1), values(degrees))
end

function _add_loop_config!(
    configs::Dict{Tuple{Vararg{Int64}},LoopConfig},
    cubic::CubicWhipGraph,
    edge_ids,
)
    key = _config_key(edge_ids)
    isempty(key) && return false
    haskey(configs, key) && return false

    edges = collect(key)
    _is_nonzero_loop_config(cubic, edges) || return false
    configs[key] = LoopConfig(edges, _config_vertices(cubic, edges), key)
    return true
end

function graph_loop_configs(
    cubic::CubicWhipGraph,
    ymax::Int64;
    compose::Bool=true,
    max_configs::Int64=100_000,
)
    @assert ymax >= 4
    raw_cycles = simplecycles_limited_length(cubic.graph, ymax)
    base_keys = Set{Tuple{Vararg{Int64}}}()
    configs = Dict{Tuple{Vararg{Int64}},LoopConfig}()

    for cycle in raw_cycles
        vertices = collect(Int64, cycle)
        4 <= length(vertices) <= ymax || continue
        key = _config_key(_config_cycle_edge_ids(cubic, vertices))
        push!(base_keys, key)
        _add_loop_config!(configs, cubic, key)
    end

    if compose
        base = collect(base_keys)
        frontier = collect(keys(configs))
        cursor = 1
        while cursor <= length(frontier)
            current = frontier[cursor]
            cursor += 1

            for loop_key in base
                candidate = _config_key(union(current, loop_key))
                length(candidate) <= ymax || continue
                if _add_loop_config!(configs, cubic, candidate)
                    push!(frontier, candidate)
                    length(configs) <= max_configs || error("Loop config count exceeded max_configs=$max_configs")
                end
            end
        end
    end

    return sort!(collect(values(configs)), by=config -> (length(config.edges), config.key))
end

function _config_edge_position(tn::DoubleLayerTN, site::Int64, edge_id::Int64)
    pos = findfirst(==(edge_id), tn.incident_edges[site])
    @assert pos !== nothing
    return pos
end

function _config_reduced_factor(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    site::Int64,
    selected_edges::Vector{Int64},
    site_value::Float64,
)
    factor = tn.factors[site]
    neighbors = tn.incident_neighbors[site]
    selected_positions = [_config_edge_position(tn, site, edge_id) for edge_id in selected_edges]
    reduced_dims = Tuple(size(factor, pos) for pos in selected_positions)
    reduced = zeros(Float64, reduced_dims)

    for ci in CartesianIndices(factor)
        inds = Tuple(ci)
        weight = factor[ci]
        for pos in eachindex(neighbors)
            pos in selected_positions && continue
            weight *= messages[(neighbors[pos], site)][inds[pos]]
        end
        reduced[ntuple(k -> inds[selected_positions[k]], length(selected_positions))...] += weight
    end

    if abs(site_value) > eps(Float64)
        reduced ./= site_value
    end
    return reduced
end

function _config_q_matrix(
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

function config_relative_correction(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    config::LoopConfig;
    site_values::Union{Nothing,Vector{Float64}}=nothing,
)
    values = site_values === nothing ? bethe_contraction_value(tn, messages)[2] : site_values
    endpoint_indices = Dict{Tuple{Int64,Int64},Index}()

    for edge_id in config.edges
        u, v = tn.edge_endpoints[edge_id]
        q = tn.edge_dims[edge_id]
        endpoint_indices[(u, edge_id)] = Index(q, "cfg,u=$u,e=$edge_id")
        endpoint_indices[(v, edge_id)] = Index(q, "cfg,u=$v,e=$edge_id")
    end

    selected = Set(config.edges)
    contraction = ITensor(1.0)

    for site in config.vertices
        site_edges = [edge_id for edge_id in tn.incident_edges[site] if edge_id in selected]
        reduced = _config_reduced_factor(tn, messages, site, site_edges, values[site])
        inds = [endpoint_indices[(site, edge_id)] for edge_id in site_edges]
        contraction *= ITensor(reduced, inds...)
    end

    for edge_id in config.edges
        u, v = tn.edge_endpoints[edge_id]
        qmat = _config_q_matrix(tn, messages, u, v, edge_id)
        contraction *= ITensor(qmat, endpoint_indices[(u, edge_id)], endpoint_indices[(v, edge_id)])
    end

    return scalar(contraction)
end

function config_loop_contraction_value(
    tn::DoubleLayerTN,
    cubic::CubicWhipGraph;
    ymax::Int64=4,
    lmax::Union{Nothing,Int64}=nothing,
    compose::Bool=true,
    max_configs::Int64=100_000,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    cutoff = lmax === nothing ? ymax : lmax
    bp = run_bp(tn; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=true)
    configs = graph_loop_configs(cubic, cutoff; compose=compose, max_configs=max_configs)
    corrections = [
        config_relative_correction(tn, bp.messages, config; site_values=bp.site_values)
        for config in configs
    ]
    return LoopConfigBPResult(bp, configs, corrections, bp.value * (1 + sum(corrections)))
end

function config_loop_norm(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    ymax::Int64=4,
    lmax::Union{Nothing,Int64}=nothing,
    compose::Bool=true,
    max_configs::Int64=100_000,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    tn = double_layer_tn(psi, cubic)
    return config_loop_contraction_value(
        tn,
        cubic;
        ymax=ymax,
        lmax=lmax,
        compose=compose,
        max_configs=max_configs,
        maxiter=maxiter,
        tol=tol,
        damping=damping,
    )
end

function config_loop_zz_projector(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    i::Int64,
    j::Int64;
    ymax::Int64=4,
    lmax::Union{Nothing,Int64}=nothing,
    compose::Bool=true,
    max_configs::Int64=100_000,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    denom = config_loop_norm(
        psi,
        cubic;
        ymax=ymax,
        lmax=lmax,
        compose=compose,
        max_configs=max_configs,
        maxiter=maxiter,
        tol=tol,
        damping=damping,
    )
    kwargs = (; ymax=ymax, lmax=lmax, compose=compose, max_configs=max_configs, maxiter=maxiter, tol=tol, damping=damping)
    pp_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:up), j => projector_operator_matrix(:up)))
    pm_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:up), j => projector_operator_matrix(:down)))
    mp_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:down), j => projector_operator_matrix(:up)))
    mm_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:down), j => projector_operator_matrix(:down)))

    pp = config_loop_contraction_value(pp_tn, cubic; kwargs...)
    pm = config_loop_contraction_value(pm_tn, cubic; kwargs...)
    mp = config_loop_contraction_value(mp_tn, cubic; kwargs...)
    mm = config_loop_contraction_value(mm_tn, cubic; kwargs...)
    zz = (pp.value - pm.value - mp.value + mm.value) / denom.value
    return zz, (upup=pp, updown=pm, downup=mp, downdown=mm), denom
end

function config_loop_sink_zz_projector(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    ymax::Int64=4,
    lmax::Union{Nothing,Int64}=nothing,
    compose::Bool=true,
    max_configs::Int64=100_000,
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    i, j = last_sink_bond(cubic)
    return config_loop_zz_projector(
        psi,
        cubic,
        i,
        j;
        ymax=ymax,
        lmax=lmax,
        compose=compose,
        max_configs=max_configs,
        maxiter=maxiter,
        tol=tol,
        damping=damping,
    )
end
