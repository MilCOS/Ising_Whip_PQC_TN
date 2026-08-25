using LinearAlgebra
using Graphs

include("tns_funcs.jl")

struct DoubleLayerTN
    graph::SimpleGraph{Int64}
    factors::Vector{Array{Float64}}
    incident_edges::Vector{Vector{Int64}}
    incident_neighbors::Vector{Vector{Int64}}
    edge_dims::Vector{Int64}
    edge_endpoints::Vector{Tuple{Int64,Int64}}
    edge_lookup::Dict{Tuple{Int64,Int64},Int64}
end

struct BPResult
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}}
    converged::Bool
    iterations::Int64
    residual::Float64
    value::Float64
    site_values::Vector{Float64}
    edge_values::Vector{Float64}
end

z_operator_matrix() = [1.0 0.0; 0.0 -1.0]
identity_operator_matrix() = Matrix{Float64}(I, 2, 2)
projector_operator_matrix(state::Symbol) = state == :up ? [1.0 0.0; 0.0 0.0] :
                                           state == :down ? [0.0 0.0; 0.0 1.0] :
                                           error("Unknown projector state: $state")

function _sorted_edge(u::Int64, v::Int64)
    return u < v ? (u, v) : (v, u)
end

function _edge_lookup(cubic::CubicWhipGraph)
    return Dict(_sorted_edge(u, v) => edge_id for (edge_id, (u, v)) in enumerate(cubic.bonds))
end

function _incident_data(cubic::CubicWhipGraph)
    nbits = length(cubic.coords)
    incident_edges = [Int64[] for _ in 1:nbits]
    incident_neighbors = [Int64[] for _ in 1:nbits]
    for (edge_id, (u, v)) in enumerate(cubic.bonds)
        push!(incident_edges[u], edge_id)
        push!(incident_neighbors[u], v)
        push!(incident_edges[v], edge_id)
        push!(incident_neighbors[v], u)
    end
    return incident_edges, incident_neighbors
end

function _site_index(tensor::ITensor)
    sites = filter(ind -> hastags(ind, "Site"), collect(inds(tensor)))
    @assert length(sites) == 1
    return first(sites)
end

function _edge_link_index(tensor::ITensor, edge_id::Int64)
    links = filter(ind -> hastags(ind, "Link") && hastags(ind, "e=$(edge_id)"), collect(inds(tensor)))
    @assert length(links) == 1
    return first(links)
end

function local_double_factor(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    site::Int64,
    edge_ids::Vector{Int64},
    op::Matrix{Float64}=identity_operator_matrix(),
)
    tensor = psi[site]
    site_ind = _site_index(tensor)
    link_inds = [_edge_link_index(tensor, edge_id) for edge_id in edge_ids]
    link_dims = dim.(link_inds)
    arr = Array(tensor, site_ind, link_inds...)
    qdims = link_dims .^ 2
    factor = zeros(Float64, Tuple(qdims))

    for ci in CartesianIndices(factor)
        combined = Tuple(ci)
        ket_inds = ntuple(k -> rem(combined[k] - 1, link_dims[k]) + 1, length(edge_ids))
        bra_inds = ntuple(k -> div(combined[k] - 1, link_dims[k]) + 1, length(edge_ids))
        val = 0.0
        for p in 1:2, q in 1:2
            val += arr[q, ket_inds...] * op[p, q] * arr[p, bra_inds...]
        end
        factor[ci] = val
    end
    return factor
end

function double_layer_tn(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    operators::Dict{Int64,Matrix{Float64}}=Dict{Int64,Matrix{Float64}}(),
)
    incident_edges, incident_neighbors = _incident_data(cubic)
    factors = Array{Float64}[]
    for site in eachindex(psi)
        op = get(operators, site, identity_operator_matrix())
        push!(factors, local_double_factor(psi, cubic, site, incident_edges[site], op))
    end
    edge_dims = Int64[]
    for (edge_id, (u, _)) in enumerate(cubic.bonds)
        link = _edge_link_index(psi[u], edge_id)
        push!(edge_dims, dim(link)^2)
    end
    return DoubleLayerTN(
        cubic.graph,
        factors,
        incident_edges,
        incident_neighbors,
        edge_dims,
        cubic.bonds,
        _edge_lookup(cubic),
    )
end

function initialize_messages(tn::DoubleLayerTN)
    messages = Dict{Tuple{Int64,Int64},Vector{Float64}}()
    for (edge_id, (u, v)) in enumerate(tn.edge_endpoints)
        q = tn.edge_dims[edge_id]
        messages[(u, v)] = fill(1.0 / q, q)
        messages[(v, u)] = fill(1.0 / q, q)
    end
    return messages
end

function _normalize_message!(message::Vector{Float64})
    scale = sum(abs, message)
    if scale <= eps(Float64)
        fill!(message, 1.0 / length(message))
    else
        message ./= scale
    end
    return message
end

function pair_normalize_messages!(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
)
    for (u, v) in tn.edge_endpoints
        overlap = dot(messages[(u, v)], messages[(v, u)])
        if overlap > eps(Float64)
            scale = sqrt(overlap)
            messages[(u, v)] ./= scale
            messages[(v, u)] ./= scale
        end
    end
    return messages
end

function _message_residual(
    old_messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    new_messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
)
    residual = 0.0
    for key in keys(old_messages)
        residual = max(residual, norm(new_messages[key] - old_messages[key], Inf))
    end
    return residual
end

function _contract_factor(
    factor::Array{Float64},
    neighbors::Vector{Int64},
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    site::Int64;
    skip_pos::Int64=0,
)
    if skip_pos == 0
        total = 0.0
        for ci in CartesianIndices(factor)
            inds = Tuple(ci)
            weight = factor[ci]
            for pos in eachindex(neighbors)
                weight *= messages[(neighbors[pos], site)][inds[pos]]
            end
            total += weight
        end
        return total
    else
        q = size(factor, skip_pos)
        out = zeros(Float64, q)
        for ci in CartesianIndices(factor)
            inds = Tuple(ci)
            weight = factor[ci]
            for pos in eachindex(neighbors)
                pos == skip_pos && continue
                weight *= messages[(neighbors[pos], site)][inds[pos]]
            end
            out[inds[skip_pos]] += weight
        end
        return out
    end
end

function update_bp_messages(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}};
    damping::Float64=0.5,
    pair_normalize::Bool=false,
)
    new_messages = deepcopy(messages)
    for site in vertices(tn.graph)
        neighbors = tn.incident_neighbors[site]
        factor = tn.factors[site]
        for pos in eachindex(neighbors)
            nbr = neighbors[pos]
            proposed = _contract_factor(factor, neighbors, messages, site; skip_pos=pos)
            _normalize_message!(proposed)
            old = messages[(site, nbr)]
            updated = (1 - damping) .* old .+ damping .* proposed
            _normalize_message!(updated)
            new_messages[(site, nbr)] = updated
        end
    end
    pair_normalize && pair_normalize_messages!(tn, new_messages)
    residual = _message_residual(messages, new_messages)
    return new_messages, residual
end

function bethe_contraction_value(tn::DoubleLayerTN, messages::Dict{Tuple{Int64,Int64},Vector{Float64}})
    site_values = zeros(Float64, nv(tn.graph))
    for site in vertices(tn.graph)
        site_values[site] = _contract_factor(
            tn.factors[site],
            tn.incident_neighbors[site],
            messages,
            site;
            skip_pos=0,
        )
    end

    edge_values = zeros(Float64, length(tn.edge_endpoints))
    for (edge_id, (u, v)) in enumerate(tn.edge_endpoints)
        edge_values[edge_id] = dot(messages[(u, v)], messages[(v, u)])
    end

    value = prod(site_values) / prod(edge_values)
    return value, site_values, edge_values
end

function run_bp(
    tn::DoubleLayerTN;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
    pair_normalize::Bool=false,
    messages::Union{Nothing,Dict{Tuple{Int64,Int64},Vector{Float64}}}=nothing,
)
    current = messages === nothing ? initialize_messages(tn) : deepcopy(messages)
    pair_normalize && pair_normalize_messages!(tn, current)
    residual = Inf
    converged = false
    iter = 0
    for n in 1:maxiter
        iter = n
        current, residual = update_bp_messages(tn, current; damping=damping, pair_normalize=pair_normalize)
        if residual < tol
            converged = true
            break
        end
    end
    value, site_values, edge_values = bethe_contraction_value(tn, current)
    return BPResult(current, converged, iter, residual, value, site_values, edge_values)
end

function bp_network_value(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    operators::Dict{Int64,Matrix{Float64}}=Dict{Int64,Matrix{Float64}}(),
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
    pair_normalize::Bool=false,
)
    tn = double_layer_tn(psi, cubic; operators=operators)
    return run_bp(tn; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=pair_normalize)
end
