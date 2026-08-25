if !isdefined(Main, :DoubleLayerTN)
    include("bp_contraction.jl")
end
if !isdefined(Main, :bp_projector_probability)
    include("bp_projector.jl")
end

struct LoopBPResult
    bp::BPResult
    plaquette_corrections::Vector{Float64}
    value::Float64
end

function _edge_id(cubic::CubicWhipGraph, u::Int64, v::Int64)
    return _edge_lookup(cubic)[_sorted_edge(u, v)]
end

function cubic_square_plaquettes(cubic::CubicWhipGraph)
    lx, ly, lz = cubic.dims
    plaquettes = NamedTuple[]

    for x in 0:lx-2, y in 0:ly-2, z in 0:lz-1
        vertices = (
            cubic.coord_to_idx[(x, y, z)],
            cubic.coord_to_idx[(x + 1, y, z)],
            cubic.coord_to_idx[(x + 1, y + 1, z)],
            cubic.coord_to_idx[(x, y + 1, z)],
        )
        edges = (
            _edge_id(cubic, vertices[1], vertices[2]),
            _edge_id(cubic, vertices[2], vertices[3]),
            _edge_id(cubic, vertices[3], vertices[4]),
            _edge_id(cubic, vertices[4], vertices[1]),
        )
        push!(plaquettes, (plane=:xy, coord=(x, y, z), vertices=vertices, edges=edges))
    end

    for x in 0:lx-2, y in 0:ly-1, z in 0:lz-2
        vertices = (
            cubic.coord_to_idx[(x, y, z)],
            cubic.coord_to_idx[(x + 1, y, z)],
            cubic.coord_to_idx[(x + 1, y, z + 1)],
            cubic.coord_to_idx[(x, y, z + 1)],
        )
        edges = (
            _edge_id(cubic, vertices[1], vertices[2]),
            _edge_id(cubic, vertices[2], vertices[3]),
            _edge_id(cubic, vertices[3], vertices[4]),
            _edge_id(cubic, vertices[4], vertices[1]),
        )
        push!(plaquettes, (plane=:xz, coord=(x, y, z), vertices=vertices, edges=edges))
    end

    for x in 0:lx-1, y in 0:ly-2, z in 0:lz-2
        vertices = (
            cubic.coord_to_idx[(x, y, z)],
            cubic.coord_to_idx[(x, y + 1, z)],
            cubic.coord_to_idx[(x, y + 1, z + 1)],
            cubic.coord_to_idx[(x, y, z + 1)],
        )
        edges = (
            _edge_id(cubic, vertices[1], vertices[2]),
            _edge_id(cubic, vertices[2], vertices[3]),
            _edge_id(cubic, vertices[3], vertices[4]),
            _edge_id(cubic, vertices[4], vertices[1]),
        )
        push!(plaquettes, (plane=:yz, coord=(x, y, z), vertices=vertices, edges=edges))
    end

    return plaquettes
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
    loop_edge_ids::Tuple{Int64,Int64},
)
    factor = tn.factors[site]
    neighbors = tn.incident_neighbors[site]
    pos1 = _edge_position(tn, site, loop_edge_ids[1])
    pos2 = _edge_position(tn, site, loop_edge_ids[2])
    reduced = zeros(Float64, size(factor, pos1), size(factor, pos2))

    for ci in CartesianIndices(factor)
        inds = Tuple(ci)
        weight = factor[ci]
        for pos in eachindex(neighbors)
            if pos == pos1 || pos == pos2
                continue
            end
            weight *= messages[(neighbors[pos], site)][inds[pos]]
        end
        reduced[inds[pos1], inds[pos2]] += weight
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

function plaquette_loop_correction(
    tn::DoubleLayerTN,
    messages::Dict{Tuple{Int64,Int64},Vector{Float64}},
    plaquette,
)
    v1, v2, v3, v4 = plaquette.vertices
    e12, e23, e34, e41 = plaquette.edges

    r1 = _reduced_factor_for_loop_edges(tn, messages, v1, (e41, e12))
    r2 = _reduced_factor_for_loop_edges(tn, messages, v2, (e12, e23))
    r3 = _reduced_factor_for_loop_edges(tn, messages, v3, (e23, e34))
    r4 = _reduced_factor_for_loop_edges(tn, messages, v4, (e34, e41))

    q12 = _q_matrix(tn, messages, v1, v2, e12)
    q23 = _q_matrix(tn, messages, v2, v3, e23)
    q34 = _q_matrix(tn, messages, v3, v4, e34)
    q41 = _q_matrix(tn, messages, v4, v1, e41)

    dims = (
        tn.edge_dims[e12],
        tn.edge_dims[e12],
        tn.edge_dims[e23],
        tn.edge_dims[e23],
        tn.edge_dims[e34],
        tn.edge_dims[e34],
        tn.edge_dims[e41],
        tn.edge_dims[e41],
    )
    total = 0.0
    for ci in CartesianIndices(dims)
        x12_1, x12_2, x23_2, x23_3, x34_3, x34_4, x41_4, x41_1 = Tuple(ci)
        total += r1[x41_1, x12_1] *
                 r2[x12_2, x23_2] *
                 r3[x23_3, x34_3] *
                 r4[x34_4, x41_4] *
                 q12[x12_1, x12_2] *
                 q23[x23_2, x23_3] *
                 q34[x34_3, x34_4] *
                 q41[x41_4, x41_1]
    end
    return total
end

function loop_contraction_value(
    tn::DoubleLayerTN,
    cubic::CubicWhipGraph;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    bp = run_bp(tn; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=true)
    corrections = [
        plaquette_loop_correction(tn, bp.messages, plaquette)
        for plaquette in cubic_square_plaquettes(cubic)
    ]
    return LoopBPResult(bp, corrections, bp.value + sum(corrections))
end

function loop_norm(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    tn = double_layer_tn(psi, cubic)
    return loop_contraction_value(tn, cubic; maxiter=maxiter, tol=tol, damping=damping)
end

function loop_zz(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    i::Int64,
    j::Int64;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    denom = loop_norm(psi, cubic; maxiter=maxiter, tol=tol, damping=damping)
    numer_tn = double_layer_tn(psi, cubic; operators=Dict(i => z_operator_matrix(), j => z_operator_matrix()))
    numer = loop_contraction_value(numer_tn, cubic; maxiter=maxiter, tol=tol, damping=damping)
    return numer.value / denom.value, numer, denom
end

function loop_sink_zz(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    i, j = last_sink_bond(cubic)
    return loop_zz(psi, cubic, i, j; maxiter=maxiter, tol=tol, damping=damping)
end

function loop_zz_projector(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph,
    i::Int64,
    j::Int64;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    denom = loop_norm(psi, cubic; maxiter=maxiter, tol=tol, damping=damping)
    pp_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:up), j => projector_operator_matrix(:up)))
    pm_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:up), j => projector_operator_matrix(:down)))
    mp_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:down), j => projector_operator_matrix(:up)))
    mm_tn = double_layer_tn(psi, cubic; operators=Dict(i => projector_operator_matrix(:down), j => projector_operator_matrix(:down)))

    pp = loop_contraction_value(pp_tn, cubic; maxiter=maxiter, tol=tol, damping=damping)
    pm = loop_contraction_value(pm_tn, cubic; maxiter=maxiter, tol=tol, damping=damping)
    mp = loop_contraction_value(mp_tn, cubic; maxiter=maxiter, tol=tol, damping=damping)
    mm = loop_contraction_value(mm_tn, cubic; maxiter=maxiter, tol=tol, damping=damping)
    zz = (pp.value - pm.value - mp.value + mm.value) / denom.value
    return zz, (upup=pp, updown=pm, downup=mp, downdown=mm), denom
end

function loop_sink_zz_projector(
    psi::Vector{ITensor},
    cubic::CubicWhipGraph;
    maxiter::Int64=500,
    tol::Float64=1e-10,
    damping::Float64=0.5,
)
    i, j = last_sink_bond(cubic)
    return loop_zz_projector(psi, cubic, i, j; maxiter=maxiter, tol=tol, damping=damping)
end
