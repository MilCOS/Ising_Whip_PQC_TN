using CairoMakie
using JLD2

set_theme!(Theme(
    fonts = (
        regular = "Times New Roman",
        bold = "Times New Roman Bold",
        italic = "Times New Roman Italic",
        bold_italic = "Times New Roman Bold Italic",
    ),
    Axis = (
        xgridvisible=false,
        ygridvisible=false,
    ),
    Legend = (
        patchsize=(30,30),
        framevisible=false,
    ),
))

function usage()
  println("""
Usage:
  julia --project=. two_dimensional/inspect_noise.jl --input=PATH/TO/FILE.jld2

The output PNG is written beside the input file with the same filename stem.
Both '--input=FILE' and '--input FILE' are accepted.
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

function analytic_edge_x(theta)
  cosine2 = cos(2theta)
  cosine2 <= 0 && return 0.0
  return cosine2 / cos(theta)
end

function output_path(input_path)
  stem, extension = splitext(input_path)
  lowercase(extension) == ".jld2" || @warn "Input filename does not end in .jld2" input_path
  return stem * ".png"
end

function inspect_noise(input_path)
  data = load(input_path)
  required = [
    "saved_thetas",
    "saved_sigmas",
    "saved_zz_mean",
    "saved_zz_sem",
    "saved_edge_operators",
    "saved_edge_mean",
    "saved_edge_sem",
  ]
  require_keys(data, required)

  thetas = Vector{Float64}(data["saved_thetas"])
  sigmas = Vector{Float64}(data["saved_sigmas"])
  zz_mean = data["saved_zz_mean"]
  zz_sem = data["saved_zz_sem"]
  edge_operators = String.(data["saved_edge_operators"])
  edge_mean = data["saved_edge_mean"]
  edge_sem = data["saved_edge_sem"]

  size(zz_mean) == (length(sigmas), length(thetas)) ||
    error("saved_zz_mean has incompatible dimensions $(size(zz_mean))")
  size(zz_sem) == size(zz_mean) || error("saved_zz_sem dimensions do not match saved_zz_mean")
  size(edge_mean) == (length(edge_operators), length(sigmas), length(thetas)) ||
    error("saved_edge_mean has incompatible dimensions $(size(edge_mean))")
  size(edge_sem) == size(edge_mean) ||
    error("saved_edge_sem dimensions do not match saved_edge_mean")

  noperators = length(edge_operators)
  figure = Figure(size=(520 * (noperators + 1), 430), fontsize=17)
  axes = Axis[]
  push!(axes, Axis(figure[1, 1]; xlabel="theta / pi", ylabel="terminal <ZZ>"))
  for (operator_id, operator) in enumerate(edge_operators)
    push!(axes, Axis(figure[1, operator_id + 1]; xlabel="theta / pi",
                    ylabel="lower-boundary <$operator>"))
  end
  
  colors = cgrad(:heat, length(sigmas)+1; categorical=true)[2:end]
  colors = ["#$s" for s in ["fed9b7","84a98c","f07167","a44a3f"]]
  xvalues = thetas ./ pi
  for sigma_id in eachindex(sigmas)
    label = "sigma/pi=$(round(sigmas[sigma_id] / pi; digits=4))"
    color = colors[sigma_id]
    scatterlines!(axes[1], xvalues, zz_mean[sigma_id, :]; color=color, linewidth=0.9, markersize=8, markercolor=(:white,0.6), strokecolor=color, strokewidth=1.0, label=label)
    band!(axes[1], xvalues,
          zz_mean[sigma_id, :] .- zz_sem[sigma_id, :],
          zz_mean[sigma_id, :] .+ zz_sem[sigma_id, :]; color=(color, 0.18))
    for operator_id in eachindex(edge_operators)
      values = edge_mean[operator_id, sigma_id, :]
      errors = edge_sem[operator_id, sigma_id, :]
      scatterlines!(axes[operator_id + 1], xvalues, values; color=color, linewidth=0.9, markersize=8, markercolor=(:white,0.6), strokecolor=color, strokewidth=1.0, label=label)
      band!(axes[operator_id + 1], xvalues, values .- errors, values .+ errors;
            color=(color, 0.18))
    end
  end

  x_id = findfirst(==("X"), edge_operators)
  if !isnothing(x_id)
    analytic = analytic_edge_x.(thetas)
    p_ana = lines!(axes[x_id + 1], xvalues, analytic; color=:black, linestyle=:dash,
           linewidth=2.0, label="analytic")
    # text!(axes[x_id + 1], 0.04, 0.96; text="analytic",
          # space=:relative, color=:magenta, fontsize=14, align=(:left, :top))
    translate!(p_ana,0,0,-100)
  end
  for axis in axes
    axislegend(axis; position=:rt, framevisible=false, labelsize=11)
  end

  destination = output_path(input_path)
  save(destination, figure)
  println("Loaded data: $input_path")
  println("Saved plot: $destination")
  return destination
end

if abspath(PROGRAM_FILE) == @__FILE__
  inspect_noise(parse_input(ARGS))
end
