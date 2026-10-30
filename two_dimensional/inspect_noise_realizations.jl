using CairoMakie
using JLD2

function usage()
  println("""
Usage:
  julia --project=. two_dimensional/inspect_noise_realizations.jl \
    --input=PATH/TO/FILE.jld2

Each disorder realization is drawn in its own subplot. Separate PNG files are
created for terminal ZZ and for every saved lower-boundary operator. No disorder
average or error band is drawn. Both '--input=FILE' and '--input FILE' work.
""")
end

function parse_input(args)
  any(arg -> arg in ("-h", "--help"), args) && (usage(); exit())
  input = nothing
  index = 1
  while index <= length(args)
    argument = args[index]
    if startswith(argument, "--input=")
      input = split(argument, "="; limit=2)[2]
    elseif argument == "--input"
      index == length(args) && error("--input requires a filename")
      index += 1
      input = args[index]
    else
      error("Unknown argument '$argument'. Use --help.")
    end
    index += 1
  end
  isnothing(input) && error("Missing required --input filename. Use --help.")
  input_path = abspath(input)
  isfile(input_path) || error("Input file does not exist: $input_path")
  return input_path
end

function require_keys(data, keys)
  missing_keys = filter(key -> !haskey(data, key), keys)
  isempty(missing_keys) || error("Input file is missing datasets: $(join(missing_keys, ", "))")
  return nothing
end

function safe_name(name)
  return lowercase(replace(name, r"[^A-Za-z0-9]+" => "_"))
end

function realization_grid(nrealizations)
  ncolumns = ceil(Int, sqrt(nrealizations))
  nrows = ceil(Int, nrealizations / ncolumns)
  return nrows, ncolumns
end

function plot_realizations(thetas, sigmas, samples, ylabel, destination)
  nsigma, ntheta, nrealizations = size(samples)
  nsigma == length(sigmas) || error("Noise-strength dimension does not match saved_sigmas")
  ntheta == length(thetas) || error("Theta dimension does not match saved_thetas")

  nrows, ncolumns = realization_grid(nrealizations)
  figure = Figure(size=(360 * ncolumns, 275 * nrows), fontsize=15)
  colors = cgrad(:viridis, nsigma; categorical=true)
  xvalues = thetas ./ pi
  axes = Axis[]

  for realization in 1:nrealizations
    row = div(realization - 1, ncolumns) + 1
    column = mod(realization - 1, ncolumns) + 1
    axis = Axis(figure[row, column];
                title="realization $realization",
                xlabel="theta / pi",
                ylabel=ylabel)
    push!(axes, axis)
    for sigma_id in eachindex(sigmas)
      label = "sigma/pi=$(round(sigmas[sigma_id] / pi; digits=4))"
      lines!(axis, xvalues, view(samples, sigma_id, :, realization);
             color=colors[sigma_id], linewidth=2.2, label=label)
    end
  end

  linkxaxes!(axes...)
  linkyaxes!(axes...)
  axislegend(first(axes); position=:rt, framevisible=false, labelsize=10)
  save(destination, figure)
  return destination
end

function inspect_noise_realizations(input_path)
  data = load(input_path)
  required = [
    "saved_thetas",
    "saved_sigmas",
    "saved_zz_samples",
    "saved_edge_operators",
    "saved_edge_samples",
  ]
  require_keys(data, required)

  thetas = Vector{Float64}(data["saved_thetas"])
  sigmas = Vector{Float64}(data["saved_sigmas"])
  zz_samples = data["saved_zz_samples"]
  edge_operators = String.(data["saved_edge_operators"])
  edge_samples = data["saved_edge_samples"]

  ndims(zz_samples) == 3 || error("saved_zz_samples must be a 3-dimensional array")
  expected_edge_size = (length(edge_operators), size(zz_samples)...)
  size(edge_samples) == expected_edge_size ||
    error("saved_edge_samples has dimensions $(size(edge_samples)); expected $expected_edge_size")

  stem, extension = splitext(input_path)
  lowercase(extension) == ".jld2" || @warn "Input filename does not end in .jld2" input_path
  outputs = String[]

  zz_path = stem * "_realizations_terminal_zz.png"
  push!(outputs, plot_realizations(thetas, sigmas, zz_samples, "terminal <ZZ>", zz_path))

  for (operator_id, operator) in enumerate(edge_operators)
    operator_path = stem * "_realizations_edge_$(safe_name(operator)).png"
    operator_samples = view(edge_samples, operator_id, :, :, :)
    push!(outputs,
          plot_realizations(thetas, sigmas, operator_samples,
                            "lower-boundary <$operator>", operator_path))
  end

  println("Loaded data: $input_path")
  for path in outputs
    println("Saved plot: $path")
  end
  return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
  inspect_noise_realizations(parse_input(ARGS))
end
