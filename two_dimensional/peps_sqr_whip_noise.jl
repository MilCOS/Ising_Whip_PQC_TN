using ITensors
using ITensorMPS
using JLD2
using Printf
using Random
using Statistics

# contract_twopeps_bdr_mps currently reads this value as a global.
global min_chi = 4

include(joinpath(@__DIR__, "peps_funcs.jl"))

const Coord = Tuple{Int,Int}
const DirectedEdge = Tuple{Coord,Coord}

Base.@kwdef struct NoiseConfig
  L::Int = 4
  chi::Int = 32
  min_chi::Int = 4
  cutoff::Float64 = 1e-12
  thetas::Vector{Float64} = collect(range(0.0, pi / 2; length=41))
  sigmas::Vector{Float64} = pi .* [0.0, 0.01, 0.025, 0.05]
  nrealizations::Int = 20
  selector::Symbol = :single
  selected_edges::Vector{DirectedEdge} = DirectedEdge[]
  correlation::Symbol = :independent
  distribution::Symbol = :gaussian
  edge_operators::Vector{String} = ["X"]
  seed::Int = 1234
  output_dir::String = joinpath(@__DIR__, "data", "noise")
end

function usage()
  println("""
Usage:
  julia --project=. two_dimensional/peps_sqr_whip_noise.jl [options]

Options are passed as --name=value:
  --L=4                         lattice side length
  --chi=32                      PEPS/boundary-MPS maximum bond dimension
  --min-chi=4                   boundary-MPS minimum bond dimension
  --cutoff=1e-12                tensor truncation cutoff
  --theta-min-pi=0              minimum theta divided by pi
  --theta-max-pi=1.0            maximum theta divided by pi
  --ntheta=41                   number of theta points
  --sigmas-pi=0,0.01,0.025,0.05
                                additive angle-noise standard deviations / pi
  --samples=20                  number of disorder realizations
  --selector=single             single, explicit, or all
  --gates="4,3>4,4"             directed coordinate pairs separated by ';'
  --correlation=independent     independent or global noise on selected gates
  --distribution=gaussian       gaussian or uniform (both have unit variance)
  --edge-ops=X                  comma-separated local operators, e.g. X,Z
  --seed=1234                   random seed
  --output=PATH                 output directory

For selector=single, the default gate is (L,L-1)>(L,L). For selector=explicit,
--gates is required. Additive noise is applied as alpha_g -> alpha_g + sigma*xi.
The same xi realization is reused for every theta and every sigma.
""")
end

function parse_edge(text::AbstractString)::DirectedEdge
  endpoints = split(strip(text), ">")
  length(endpoints) == 2 || error("Invalid directed edge '$text'; use x1,y1>x2,y2")
  parse_coord(value) = begin
    xy = parse.(Int, split(strip(value), ","))
    length(xy) == 2 || error("Invalid coordinate '$value'")
    (xy[1], xy[2])
  end
  return (parse_coord(endpoints[1]), parse_coord(endpoints[2]))
end

function parse_edges(text::AbstractString)::Vector{DirectedEdge}
  isempty(strip(text)) && return DirectedEdge[]
  return parse_edge.(split(text, ";"))
end

function parse_options(args)::NoiseConfig
  any(arg -> arg in ("-h", "--help"), args) && (usage(); exit())
  options = Dict{String,String}()
  for arg in args
    startswith(arg, "--") || error("Unknown positional argument '$arg'. Use --help.")
    pair = split(arg[3:end], "="; limit=2)
    length(pair) == 2 || error("Option '$arg' must have the form --name=value")
    options[pair[1]] = pair[2]
  end

  getoption(name, default) = get(options, name, string(default))
  theta_min = parse(Float64, getoption("theta-min-pi", 0.0)) * pi
  theta_max = parse(Float64, getoption("theta-max-pi", 1.0)) * pi
  ntheta = parse(Int, getoption("ntheta", 41))
  sigma_text = getoption("sigmas-pi", "0,0.01,0.025,0.05")
  sigmas = pi .* parse.(Float64, split(sigma_text, ","))
  edge_ops = uppercase.(strip.(split(getoption("edge-ops", "X"), ",")))
  selected_edges = parse_edges(getoption("gates", ""))

  config = NoiseConfig(
    L=parse(Int, getoption("L", 4)),
    chi=parse(Int, getoption("chi", 32)),
    min_chi=parse(Int, getoption("min-chi", 4)),
    cutoff=parse(Float64, getoption("cutoff", 1e-12)),
    thetas=collect(range(theta_min, theta_max; length=ntheta)),
    sigmas=sigmas,
    nrealizations=parse(Int, getoption("samples", 20)),
    selector=Symbol(lowercase(getoption("selector", "single"))),
    selected_edges=selected_edges,
    correlation=Symbol(lowercase(getoption("correlation", "independent"))),
    distribution=Symbol(lowercase(getoption("distribution", "gaussian"))),
    edge_operators=edge_ops,
    seed=parse(Int, getoption("seed", 1234)),
    output_dir=abspath(getoption("output", joinpath(@__DIR__, "data", "noise"))),
  )
  validate(config)
  return config
end

function validate(config::NoiseConfig)
  config.L >= 2 || error("L must be at least 2")
  config.chi >= 1 || error("chi must be positive")
  1 <= config.min_chi <= config.chi || error("min_chi must satisfy 1 <= min_chi <= chi")
  config.cutoff >= 0 || error("cutoff must be nonnegative")
  isempty(config.thetas) && error("At least one theta is required")
  isempty(config.sigmas) && error("At least one sigma is required")
  any(sigma -> sigma < 0.0, config.sigmas) && error("Noise strengths must be nonnegative")
  config.nrealizations >= 1 || error("At least one realization is required")
  config.selector in (:single, :explicit, :all) ||
    error("selector must be single, explicit, or all")
  config.correlation in (:independent, :global) ||
    error("correlation must be independent or global")
  config.distribution in (:gaussian, :uniform) ||
    error("distribution must be gaussian or uniform")
  isempty(config.edge_operators) && error("At least one edge operator is required")
  all(op -> op in ("X", "Z"), config.edge_operators) ||
    error("edge-ops currently supports X and Z")
  length(unique(config.edge_operators)) == length(config.edge_operators) ||
    error("edge-ops contains duplicates")
  return nothing
end

function resolved_edges(config::NoiseConfig)::Vector{DirectedEdge}
  if config.selector == :single && isempty(config.selected_edges)
    return [((config.L, config.L - 1), (config.L, config.L))]
  elseif config.selector == :single
    length(config.selected_edges) == 1 || error("single selector requires exactly one gate")
    return config.selected_edges
  elseif config.selector == :explicit
    isempty(config.selected_edges) && error("explicit selector requires --gates")
    return unique(config.selected_edges)
  else
    return DirectedEdge[]
  end
end

function selected_gate_indices(config::NoiseConfig, gates::GateNet)
  config.selector == :all && return collect(1:gates.Ngates)
  wanted_edges = resolved_edges(config)
  wanted_indices = [(coor2idx(edge[1], config.L), coor2idx(edge[2], config.L)) for edge in wanted_edges]
  selected = Int[]
  for edge in wanted_indices
    gate_id = findfirst(==(edge), gates.indices)
    isnothing(gate_id) && error("Directed edge $edge is not a gate in the L=$(config.L) whip circuit")
    push!(selected, gate_id)
  end
  return unique(selected)
end

function lower_boundary_sites(L::Int)
  # The two outgoing sides of the DAG nearest the sink, excluding the sink (L,L).
  coords = vcat([(L, y) for y in 1:(L - 1)], [(x, L) for x in 1:(L - 1)])
  @assert length(coords) == 2L - 2
  return coords, coor2idx.(coords, L)
end

function terminal_zz_sites(L::Int)
  coords = ((L, L - 1), (L, L))
  return coords, (coor2idx(coords[1], L), coor2idx(coords[2], L))
end

function expectation_one_site(psi, sites, site::Int, operator::String, normalization, chi, cutoff)
  observable_psi = deepcopy(psi)
  observable_psi[site] *= op(operator, sites[site])
  noprime!(observable_psi[site], tags="Site")
  value = contract_twopeps_bdr_mps(observable_psi, psi, chi, cutoff; plev=0)
  return real(value / normalization)
end

function expectation_zz(psi, sites, pair::Tuple{Int,Int}, normalization, chi, cutoff)
  observable_psi = deepcopy(psi)
  for site in pair
    observable_psi[site] *= op("Z", sites[site])
    noprime!(observable_psi[site], tags="Site")
  end
  value = contract_twopeps_bdr_mps(observable_psi, psi, chi, cutoff; plev=0)
  return real(value / normalization)
end

function standard_noise(config::NoiseConfig, nselected::Int)
  rng = MersenneTwister(config.seed)
  ncolumns = config.correlation == :global ? 1 : nselected
  noise = if config.distribution == :gaussian
    randn(rng, config.nrealizations, ncolumns)
  else
    sqrt(3.0) .* (2.0 .* rand(rng, config.nrealizations, ncolumns) .- 1.0)
  end
  return config.correlation == :global ? repeat(noise, 1, nselected) : noise
end

function mean_and_sem(samples; realization_dim::Int)
  average = dropdims(mean(samples; dims=realization_dim); dims=realization_dim)
  if size(samples, realization_dim) == 1
    standard_error = zeros(size(average))
  else
    standard_error = dropdims(std(samples; dims=realization_dim); dims=realization_dim) ./
                     sqrt(size(samples, realization_dim))
  end
  return average, standard_error
end

function selector_description(config, selected_edges)
  config.selector == :all && return "all"
  return join(["$(edge[1])>$(edge[2])" for edge in selected_edges], ";")
end

function run_noise_scan(config::NoiseConfig)
  validate(config)
  global min_chi = config.min_chi

  nsites = config.L^2
  initial_peps = twodim_sqr_peps(config.L, "+"^nsites)
  sites = get_siteinds(initial_peps)
  reference_gates = make_sqr_whip_gates_obc(config.L, first(config.thetas))
  selected_ids = selected_gate_indices(config, reference_gates)
  selected_edges = [(idx2coor(reference_gates.indices[id][1], config.L),
                     idx2coor(reference_gates.indices[id][2], config.L)) for id in selected_ids]
  noise_draws = standard_noise(config, length(selected_ids))
  boundary_coords, boundary_sites = lower_boundary_sites(config.L)
  zz_coords, zz_sites = terminal_zz_sites(config.L)

  nsigma = length(config.sigmas)
  ntheta = length(config.thetas)
  nsamples = config.nrealizations
  noperators = length(config.edge_operators)
  nboundary = length(boundary_sites)
  zz_samples = Array{Float64}(undef, nsigma, ntheta, nsamples)
  edge_local_samples = Array{Float64}(undef, noperators, nsigma, ntheta, nsamples, nboundary)

  total_states = nsigma * ntheta * nsamples
  completed = 0
  start_time = time_ns()
  println("Starting noisy PEPS scan")
  println("  L=$(config.L), chi=$(config.chi), min_chi=$(config.min_chi), cutoff=$(config.cutoff)")
  println("  selector=$(config.selector), gates=$(selector_description(config, selected_edges))")
  println("  edge operators=$(join(config.edge_operators, ',')), realizations=$(config.nrealizations)")

  for realization in 1:nsamples
    for (sigma_id, sigma) in enumerate(config.sigmas)
      if iszero(sigma) && realization > 1
        zz_samples[sigma_id, :, realization] .= zz_samples[sigma_id, :, 1]
        edge_local_samples[:, sigma_id, :, realization, :] .=
          edge_local_samples[:, sigma_id, :, 1, :]
        completed += ntheta
        @printf("  realization %d/%d, sigma/pi=%g reused clean curve (%d/%d states)\n",
                realization, nsamples, sigma / pi, completed, total_states)
        continue
      end
      for (theta_id, theta) in enumerate(config.thetas)
        psi = deepcopy(initial_peps)
        gates = make_sqr_whip_gates_obc(config.L, theta)
        gates.paras[selected_ids] .+= sigma .* noise_draws[realization, :]
        zytensors = make_tensormap_of_ZY(gates, sites)
        make_contraction_peps!(gates, zytensors, psi, config.chi, config.cutoff)

        normalization = contract_twopeps_bdr_mps(psi, psi, config.chi, config.cutoff; plev=0)
        isfinite(real(normalization)) || error("Non-finite PEPS norm at theta=$theta, sigma=$sigma")
        zz_samples[sigma_id, theta_id, realization] =
          expectation_zz(psi, sites, zz_sites, normalization, config.chi, config.cutoff)

        for operator_id in eachindex(config.edge_operators)
          operator = config.edge_operators[operator_id]
          for boundary_id in eachindex(boundary_sites)
            edge_local_samples[operator_id, sigma_id, theta_id, realization, boundary_id] =
              expectation_one_site(psi, sites, boundary_sites[boundary_id], operator,
                                   normalization, config.chi, config.cutoff)
          end
        end
        completed += 1
      end
      @printf("  realization %d/%d, sigma/pi=%g (%d/%d states)\n",
              realization, nsamples, sigma / pi, completed, total_states)
    end
  end

  edge_samples = dropdims(mean(edge_local_samples; dims=5); dims=5)
  zz_mean, zz_sem = mean_and_sem(zz_samples; realization_dim=3)
  edge_mean, edge_sem = mean_and_sem(edge_samples; realization_dim=4)
  elapsed_seconds = (time_ns() - start_time) * 1e-9

  mkpath(config.output_dir)
  gate_tag = config.selector == :all ? "all" : "g" * join(selected_ids, "-")
  base_name = "peps_whip_noise_L$(config.L)_chi$(config.chi)_$(config.selector)_" *
              "$(gate_tag)_$(config.correlation)_$(config.distribution)_" *
              "$(join(config.edge_operators, ""))_seed$(config.seed)"
  data_path = joinpath(config.output_dir, base_name * ".jld2")
  jldsave(data_path;
    saved_thetas=config.thetas,
    saved_sigmas=config.sigmas,
    saved_zz_samples=zz_samples,
    saved_zz_mean=zz_mean,
    saved_zz_sem=zz_sem,
    saved_edge_operators=config.edge_operators,
    saved_edge_local_samples=edge_local_samples,
    saved_edge_samples=edge_samples,
    saved_edge_mean=edge_mean,
    saved_edge_sem=edge_sem,
    saved_boundary_coords=boundary_coords,
    saved_terminal_zz_coords=zz_coords,
    saved_selected_gate_ids=selected_ids,
    saved_selected_edges=selected_edges,
    saved_noise_draws=noise_draws,
    noise_model="additive_angle",
    noise_correlation=String(config.correlation),
    noise_distribution=String(config.distribution),
    selector=String(config.selector),
    L=config.L,
    chi=config.chi,
    min_chi=config.min_chi,
    cutoff=config.cutoff,
    seed=config.seed,
    nrealizations=config.nrealizations,
    total_time_sec=elapsed_seconds,
  )

  println("Saved data: $data_path")
  @printf("Elapsed time: %.3f seconds\n", elapsed_seconds)
  return (; data_path, zz_mean, zz_sem, edge_mean, edge_sem)
end

if abspath(PROGRAM_FILE) == @__FILE__
  run_noise_scan(parse_options(ARGS))
end
