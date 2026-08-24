using Printf
using Graphs
using CairoMakie
using GraphMakie

"""
julia plot_cubic_graph.jl --lx 2 --ly 2 --lz 2 --theta 0.3 --out cube.png
"""

include("tns_funcs.jl")

function directed_cubic_graph(cubic::CubicWhipGraph)
    dg = SimpleDiGraph(length(cubic.coords))
    for (i, j) in cubic.gate_indices
        add_edge!(dg, i, j)
    end
    return dg
end

function projected_positions(cubic::CubicWhipGraph)
    return [
        Point2f(coor[1] + 0.42 * coor[3], coor[2] + 0.42 * coor[3])
        for coor in cubic.coords
    ]
end

function print_bond_angle_table(cubic::CubicWhipGraph)
    println("bond_id | i(Z) coord -> j(Y) coord | indeg(j) | angle")
    println("--------+--------------------------+----------+----------------")
    for (bond_id, ((i, j), angle)) in enumerate(zip(cubic.gate_indices, cubic.gate_paras))
        ci = idx2coord(i, cubic.dims)
        cj = idx2coord(j, cubic.dims)
        @printf(
            "%7d | %4d %-11s -> %4d %-11s | %8d | %.12g\n",
            bond_id,
            i,
            string(ci),
            j,
            string(cj),
            cubic.incoming_counts[j],
            angle,
        )
    end
end

function plot_cubic_whip_graph(
    lx::Int64,
    ly::Int64,
    lz::Int64,
    theta::Float64;
    outfile::String="three_dimensional/cubic_whip_graph.png",
)
    cubic = make_cubic_whip_graph(lx, ly, lz, theta)
    dg = directed_cubic_graph(cubic)
    positions = projected_positions(cubic)

    fig = Figure(size=(900, 760))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="3D cubic whip directed graph")
    hidedecorations!(ax)
    hidespines!(ax)

    node_colors = fill(:steelblue3, nv(dg))
    node_colors[cubic.source] = :seagreen3
    node_colors[cubic.sink] = :tomato
    graphplot!(
        ax,
        dg;
        layout=positions,
        arrow_show=true,
        arrow_size=16,
        node_size=24,
        node_color=node_colors,
        edge_color=:gray35,
        nlabels=string.(1:nv(dg)),
        nlabels_textsize=14,
    )

    text!(
        ax,
        positions[cubic.source];
        text="source",
        offset=(12, 12),
        fontsize=16,
        color=:seagreen4,
    )
    text!(
        ax,
        positions[cubic.sink];
        text="sink",
        offset=(12, 12),
        fontsize=16,
        color=:tomato4,
    )
    save(outfile, fig)
    return outfile, cubic
end

function main(args)
    lx,ly,lz = 2,2,2
    theta = pi / 4
    outfile = "graph.png"
    for i in eachindex(args)
        if args[i] == "--lx"
            lx = parse(Int, args[i+1])
            i += 1
        end
        if args[i] == "--ly"
            ly = parse(Int, args[i+1])
            i += 1
        end
        if args[i] == "--lz"
            lz = parse(Int, args[i+1])
            i += 1
        end
        if args[i] == "--theta"
            theta = parse(Float64, args[i+1])
            i += 1
        end
        if args[i] == "--out"
            outfile = args[i+1]
            i += 1
        end
    end
    outfile, cubic = plot_cubic_whip_graph(lx, ly, lz, theta; outfile=outfile)
    println("Saved directed graph to $outfile")
    println("source: $(cubic.source) $(idx2coord(cubic.source, cubic.dims))")
    println("sink:   $(cubic.sink) $(idx2coord(cubic.sink, cubic.dims))")
    println()
    print_bond_angle_table(cubic)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
