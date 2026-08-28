using CairoMakie

const DATA_KIND = get(ENV, "DATA_KIND", "ZZ-theta")
const SUPPORTED_DATA_KINDS = ("ZZ-theta",)

if DATA_KIND ∉ SUPPORTED_DATA_KINDS
    error("Unsupported DATA_KIND=$(DATA_KIND). Supported values: $(join(SUPPORTED_DATA_KINDS, ", ")).")
end

const SCRIPT_DIR = @__DIR__
const DATA_DIR = joinpath(SCRIPT_DIR, "data")
const FIGURE_DIR = joinpath(SCRIPT_DIR, "figures")
const TARGET_L = [4, 6]
const TARGET_LMAX = [4, 6]
const Y_LIMITS = (-0.05, 1.1)

const METHOD_ORDER = [
    :direct_bp,
    :projector_bp,
    :config_loop_bp,
]

const METHOD_LABELS = Dict(
    :direct_bp => "direct BP",
    :projector_bp => "projector BP",
    :config_loop_bp => "config loop BP",
)

const METHOD_MARKERS = Dict(
    :direct_bp => :circle,
    :projector_bp => :rect,
    :config_loop_bp => :diamond,
)

const METHOD_COLORS = Dict(
    :direct_bp => Makie.wong_colors()[1],
    :projector_bp => Makie.wong_colors()[2],
    :config_loop_bp => Makie.wong_colors()[3],
)

struct ZZRecord
    L::Int
    theta_over_pi::Float64
    lmax::Int
    values::Dict{Symbol, Float64}
    converged::Dict{Symbol, Bool}
end

function parse_converged(line::AbstractString)
    m = match(r"converged = (true|false)", line)
    isnothing(m) && return missing
    return m.captures[1] == "true"
end

function parse_value_after_equals(line::AbstractString)
    m = match(r"=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", line)
    isnothing(m) && return missing
    return parse(Float64, m.captures[1])
end

function parse_zz_file(path::AbstractString)
    filename = basename(path)
    m = match(r"^CUBE_L(\d+)_THETA([0-9.]+)pi_LMAX(\d+)_ZZ\.txt$", filename)
    isnothing(m) && return nothing

    L = parse(Int, m.captures[1])
    theta_over_pi = parse(Float64, m.captures[2])
    lmax = parse(Int, m.captures[3])

    values = Dict{Symbol, Float64}()
    convergence_terms = Dict(method => Bool[] for method in METHOD_ORDER)

    for line in eachline(path)
        c = parse_converged(line)

        if startswith(line, "direct numerator")
            !ismissing(c) && push!(convergence_terms[:direct_bp], c)
        elseif startswith(line, "denominator")
            !ismissing(c) && push!(convergence_terms[:direct_bp], c)
            !ismissing(c) && push!(convergence_terms[:projector_bp], c)
        elseif startswith(line, "projector up-up") ||
               startswith(line, "projector up-down") ||
               startswith(line, "projector down-up") ||
               startswith(line, "projector down-down")
            !ismissing(c) && push!(convergence_terms[:projector_bp], c)
        elseif startswith(line, "config projector") || startswith(line, "config denominator")
            !ismissing(c) && push!(convergence_terms[:config_loop_bp], c)
        end

        if startswith(line, "direct BP") && occursin("<ZZ>", line)
            values[:direct_bp] = parse_value_after_equals(line)
        elseif startswith(line, "projector BP") && occursin("<ZZ>", line)
            values[:projector_bp] = parse_value_after_equals(line)
        elseif startswith(line, "config loop BP") && occursin("<ZZ>", line)
            values[:config_loop_bp] = parse_value_after_equals(line)
        end
    end

    converged = Dict(method => !isempty(terms) && all(terms) for (method, terms) in convergence_terms)
    return ZZRecord(L, theta_over_pi, lmax, values, converged)
end

function load_records()
    records = ZZRecord[]
    for filename in sort(readdir(DATA_DIR))
        endswith(filename, "_ZZ.txt") || continue
        record = parse_zz_file(joinpath(DATA_DIR, filename))
        isnothing(record) || push!(records, record)
    end
    isempty(records) && error("No ZZ data files found in $(DATA_DIR).")
    return records
end

function draw_method!(ax, xs, ys, convs, method)
    color = METHOD_COLORS[method]
    marker = METHOD_MARKERS[method]

    for i in 1:(length(xs) - 1)
        linestyle = convs[i] && convs[i + 1] ? nothing : :dash
        lines!(ax, xs[i:i + 1], ys[i:i + 1]; color, linewidth=2.2, linestyle)
    end

    good = findall(identity, convs)
    bad = findall(!, convs)

    if !isempty(good)
        scatter!(ax, xs[good], ys[good]; color, marker, markersize=10)
    end
    if !isempty(bad)
        scatter!(ax, xs[bad], ys[bad];
            color=(:white, 0.0),
            strokecolor=color,
            strokewidth=2,
            marker,
            markersize=10,
        )
    end
end

function build_figure(records)
    records = filter(record -> record.L in TARGET_L && record.lmax in TARGET_LMAX, records)
    isempty(records) && error("No ZZ records matched TARGET_L=$(TARGET_L) and TARGET_LMAX=$(TARGET_LMAX).")

    L_list = TARGET_L
    lmax_list = TARGET_LMAX

    fig = Figure(size=(1180, 720), fontsize=16)
    axes = Axis[]

    for (row, lmax) in enumerate(lmax_list), (col, L) in enumerate(L_list)
        ax = Axis(
            fig[row, col],
            xlabel=row == length(lmax_list) ? "θ/π" : "",
            ylabel=col == 1 ? "<ZZ>" : "",
            title="L=$(L), lmax=$(lmax)",
            xticks=0.0:0.1:0.5,
            limits=(nothing, Y_LIMITS),
        )
        push!(axes, ax)

        panel_records = sort(
            filter(record -> record.L == L && record.lmax == lmax, records),
            by=record -> record.theta_over_pi,
        )

        for method in METHOD_ORDER
            method_records = filter(record -> haskey(record.values, method), panel_records)
            isempty(method_records) && continue

            xs = [record.theta_over_pi for record in method_records]
            ys = abs.([record.values[method] for record in method_records])
            convs = [get(record.converged, method, false) for record in method_records]
            draw_method!(ax, xs, ys, convs, method)
        end
    end

    linkxaxes!(axes...)

    method_handles = [
        [
            LineElement(color=METHOD_COLORS[method], linewidth=2.2),
            MarkerElement(
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=12,
            ),
        ]
        for method in METHOD_ORDER
    ]
    method_labels = [METHOD_LABELS[method] for method in METHOD_ORDER]

    convergence_handles = [
        [
            LineElement(color=:black, linewidth=2.2),
            MarkerElement(color=:black, marker=:circle, markersize=12),
        ],
        [
            LineElement(color=:black, linewidth=2.2, linestyle=:dash),
            MarkerElement(
                color=(:white, 0.0),
                strokecolor=:black,
                strokewidth=2,
                marker=:circle,
                markersize=12,
            ),
        ],
    ]

    Legend(fig[1:length(lmax_list), length(L_list) + 1],
        [method_handles, convergence_handles],
        [method_labels, ["converged", "not converged"]],
        ["method", "status"],
        framevisible=false,
    )

    Label(fig[0, 1:length(L_list)],
        "$(DATA_KIND): 3D whip circuit TNS-BP-Loop",
        fontsize=22,
        font=:bold,
    )

    return fig
end

records = load_records()
fig = build_figure(records)

mkpath(FIGURE_DIR)
output_path = joinpath(FIGURE_DIR, "ZZ_theta_3D_whip.png")
save(output_path, fig, px_per_unit=2)
println("Saved figure to $(output_path)")
