using ITensors, ITensorMPS, LinearAlgebra
using Graphs

include("../two_dimensional/gates.jl")

const Coord3D = NTuple{3,Int64}

"""
Data describing the open-boundary cubic whip geometry.

Coordinates are stored in the physical 0-based convention:
`source = (0, 0, 0)` and `sink = (Lx-1, Ly-1, Lz-1)`.
Site indices remain Julia 1-based, with `z` the fastest coordinate.
"""
struct CubicWhipGraph
    dims::Coord3D
    graph::SimpleGraph{Int64}
    coords::Vector{Coord3D}
    coord_to_idx::Dict{Coord3D,Int64}
    bonds::Vector{Tuple{Int64,Int64}}
    axes::Vector{Symbol}
    source::Int64
    sink::Int64
    gate_indices::Vector{Tuple{Int64,Int64}}
    gate_paras::Vector{Float64}
    incoming_counts::Vector{Int64}
end

function coord2idx(coor::Coord3D, dims::Coord3D)
    x, y, z = coor
    lx, ly, lz = dims
    @assert 0 <= x < lx
    @assert 0 <= y < ly
    @assert 0 <= z < lz
    return x * ly * lz + y * lz + z + 1
end

function idx2coord(idx::Int64, dims::Coord3D)
    lx, ly, lz = dims
    @assert 1 <= idx <= lx * ly * lz
    n = idx - 1
    x = div(n, ly * lz)
    rem_xy = rem(n, ly * lz)
    y = div(rem_xy, lz)
    z = rem(rem_xy, lz)
    return (x, y, z)
end

function cubic_coords(dims::Coord3D)
    lx, ly, lz = dims
    return [(x, y, z) for x in 0:lx-1 for y in 0:ly-1 for z in 0:lz-1]
end

function incoming_count(coor::Coord3D)
    return count(>(0), coor)
end

function _axis_step(axis::Symbol)
    if axis == :x
        return (1, 0, 0)
    elseif axis == :y
        return (0, 1, 0)
    elseif axis == :z
        return (0, 0, 1)
    else
        error("Unknown cubic axis: $axis")
    end
end

function make_cubic_whip_graph(lx::Int64, ly::Int64, lz::Int64, theta::Float64)
    @assert lx >= 1 && ly >= 1 && lz >= 1
    dims = (lx, ly, lz)
    coords = cubic_coords(dims)
    coord_to_idx = Dict(coor => coord2idx(coor, dims) for coor in coords)
    nbits = length(coords)
    graph = SimpleGraph(nbits)
    bonds = Tuple{Int64,Int64}[]
    axes = Symbol[]
    gate_data = Tuple{Int64,Int64,Float64,Coord3D,Symbol}[]
    incoming_counts = [incoming_count(coor) for coor in coords]

    for coor in coords
        src_idx = coord_to_idx[coor]
        for axis in (:x, :y, :z)
            step = _axis_step(axis)
            dst = coor .+ step
            if all(0 .<= dst .< dims)
                dst_idx = coord_to_idx[dst]
                add_edge!(graph, src_idx, dst_idx)
                push!(bonds, (src_idx, dst_idx))
                push!(axes, axis)

                indeg = incoming_counts[dst_idx]
                @assert indeg > 0
                push!(gate_data, (src_idx, dst_idx, theta / indeg, dst, axis))
            end
        end
    end

    sort!(gate_data, by=x -> (sum(x[4]), x[4][1], x[4][2], x[4][3], x[5]))
    gate_indices = [(src, dst) for (src, dst, _, _, _) in gate_data]
    gate_paras = [angle for (_, _, angle, _, _) in gate_data]

    source = coord_to_idx[(0, 0, 0)]
    sink = coord_to_idx[(lx - 1, ly - 1, lz - 1)]
    return CubicWhipGraph(
        dims,
        graph,
        coords,
        coord_to_idx,
        bonds,
        axes,
        source,
        sink,
        gate_indices,
        gate_paras,
        incoming_counts,
    )
end

function make_cubic_whip_gates(cubic::CubicWhipGraph)
    return GateNet(length(cubic.gate_paras), cubic.gate_indices, cubic.gate_paras)
end

function make_cubic_whip_gates(lx::Int64, ly::Int64, lz::Int64, theta::Float64)
    cubic = make_cubic_whip_graph(lx, ly, lz, theta)
    return make_cubic_whip_gates(cubic)
end

function make_tensormap_of_ZY(gates::GateNet, sites::Vector{Index{Int64}})
    t_mat = zeros(Float64, 4, 4)
    Tm = Array{ITensor}(undef, gates.Ngates)
    for (idx, coor) in enumerate(gates.indices)
        i, j = coor
        s_i, s_j = sites[i], sites[j]
        _tensorzy!(t_mat, gates.paras[idx])
        comb_1 = combiner(prime(s_i), prime(s_j))
        comb_2 = combiner(s_i, s_j)
        Tc = ITensor(t_mat, combinedind(comb_1), combinedind(comb_2))
        Tm[idx] = Tc * dag(comb_1) * dag(comb_2)
    end
    return Tm
end

function last_sink_bond(cubic::CubicWhipGraph)
    sink_coord = idx2coord(cubic.sink, cubic.dims)
    candidates = Coord3D[]
    for axis in (:z, :y, :x)
        step = _axis_step(axis)
        coor = sink_coord .- step
        if all(0 .<= coor .< cubic.dims)
            push!(candidates, coor)
        end
    end
    @assert !isempty(candidates)
    return (cubic.coord_to_idx[first(candidates)], cubic.sink)
end

function tns_product_state(cubic::CubicWhipGraph, st::String, sites::Vector{T}=Index{Int64}[]) where T
    nbits = length(cubic.coords)
    @assert length(st) == nbits
    links = [Index(1, "Link,e=$(i)") for i in eachindex(cubic.bonds)]
    loc_links = [Index{Int64}[] for _ in 1:nbits]
    for (edge_id, (i, j)) in enumerate(cubic.bonds)
        push!(loc_links[i], links[edge_id])
        push!(loc_links[j], links[edge_id])
    end

    loc_sites = isempty(sites) ? siteinds("Qubit", nbits) : sites
    vals_dict = Dict(
        '0' => [1.0, 0.0],
        '1' => [0.0, 1.0],
        '+' => [1.0 / sqrt(2), 1.0 / sqrt(2)],
        '-' => [1.0 / sqrt(2), -1.0 / sqrt(2)],
    )

    tensors = ITensor[]
    for i in 1:nbits
        vals = vals_dict[st[i]]
        push!(tensors, ITensor(vals, loc_sites[i], loc_links[i]...))
    end
    return tensors
end

function tns_equal_superposition(cubic::CubicWhipGraph)
    return tns_product_state(cubic, "+" ^ length(cubic.coords))
end

function get_siteinds(tns::Vector{ITensor})
    sites = Index{Int64}[]
    for tensor in tns
        append!(sites, filter(ind -> hastags(ind, "Site"), collect(inds(tensor))))
    end
    return sites
end

function tns_local_contraction!(
    psi::Vector{ITensor},
    locT2::ITensor,
    i::Int64,
    j::Int64,
    chi::Int64,
    cutoff::Float64=1e-12,
)
    old_links = filter(ind -> hastags(ind, "Link"), collect(commoninds(psi[i], psi[j])))
    @assert length(old_links) == 1
    old_link = first(old_links)

    newT2 = locT2 * (psi[i] * psi[j])
    noprime!(newT2)
    indsij = uniqueinds(psi[i], psi[j])
    U, Q = factorize(newT2, indsij; ortho="none", which_decomp="svd", maxdim=chi, cutoff=cutoff)

    new_link = commonind(U, Q)
    renamed_link = Index(dim(new_link), tags(old_link))
    replaceind!(U, new_link, renamed_link)
    replaceind!(Q, new_link, renamed_link)
    psi[i] = U
    psi[j] = Q
    return psi
end

function make_contraction_tns!(
    gates::GateNet,
    Tm::Array{ITensor},
    psi::Vector{ITensor},
    chi::Int64,
    cutoff::Float64=1e-12,
)
    for idx in 1:gates.Ngates
        i, j = gates.indices[idx]
        @assert i < j
        tns_local_contraction!(psi, Tm[idx], i, j, chi, cutoff)
    end
    return psi
end

function prepare_cubic_whip_tns(
    lx::Int64,
    ly::Int64,
    lz::Int64,
    theta::Float64;
    chi::Int64=2,
    cutoff::Float64=1e-12,
)
    cubic = make_cubic_whip_graph(lx, ly, lz, theta)
    psi = tns_equal_superposition(cubic)
    sites = get_siteinds(psi)
    gates = make_cubic_whip_gates(cubic)
    zytensors = make_tensormap_of_ZY(gates, sites)
    make_contraction_tns!(gates, zytensors, psi, chi, cutoff)
    return psi, cubic, sites
end
