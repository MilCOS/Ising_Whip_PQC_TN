using Printf

"""
julia run_bp_whip.jl \
  --lx 2 --ly 2 --lz 2 --theta 0.3 \
  --method loop --maxiter 300 --tol 1e-9 \
  --damping 0.5 --pairnorm true

Methods:
  direct     <ZZ> from a signed Z_i Z_j impurity network
  projector  <ZZ> from four positive projector networks
  loop       lmax=4 square-plaquette loop correction, using projector networks
  all        run every available method
"""

include("bp_direct.jl")
include("bp_projector.jl")
include("bp_loop.jl")

function parse_bp_args(args)
    opts = Dict{String,String}(
        "lx" => "2",
        "ly" => "2",
        "lz" => "2",
        "theta" => string(pi / 4),
        "chi" => "2",
        "cutoff" => "1e-12",
        "maxiter" => "500",
        "tol" => "1e-10",
        "damping" => "0.5",
        "pairnorm" => "false",
        "method" => "all",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            @assert haskey(opts, key) "Unknown option: $arg"
            @assert i < length(args) "Missing value for option: $arg"
            opts[key] = args[i + 1]
            i += 2
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return opts
end

parse_bool(s::String) = lowercase(s) in ("true", "t", "1", "yes", "y")

function print_bp_result(label::String, value, result::BPResult)
    @printf(
        "%-24s value = %.16g | converged = %s | iter = %d | residual = %.3e\n",
        label,
        value,
        string(result.converged),
        result.iterations,
        result.residual,
    )
end

function print_loop_result(label::String, result::LoopBPResult)
    @printf(
        "%-24s value = %.16g | bp = %.16g | sum loop = %.3e | nloops = %d | converged = %s | residual = %.3e\n",
        label,
        result.value,
        result.bp.value,
        sum(result.plaquette_corrections),
        length(result.plaquette_corrections),
        string(result.bp.converged),
        result.bp.residual,
    )
end

function run_direct_zz(psi, cubic; maxiter::Int64, tol::Float64, damping::Float64, pair_normalize::Bool)
    zz, numer, denom = bp_sink_zz(
        psi,
        cubic;
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
    print_bp_result("direct numerator", numer.value, numer)
    print_bp_result("denominator", denom.value, denom)
    @printf("%-24s <ZZ>  = %.16g\n", "direct BP", zz)
    return zz
end

function run_projector_zz(psi, cubic; maxiter::Int64, tol::Float64, damping::Float64, pair_normalize::Bool)
    zz, parts, denom = bp_sink_zz_projector(
        psi,
        cubic;
        maxiter=maxiter,
        tol=tol,
        damping=damping,
        pair_normalize=pair_normalize,
    )
    print_bp_result("projector up-up", parts.upup.value, parts.upup)
    print_bp_result("projector up-down", parts.updown.value, parts.updown)
    print_bp_result("projector down-up", parts.downup.value, parts.downup)
    print_bp_result("projector down-down", parts.downdown.value, parts.downdown)
    print_bp_result("denominator", denom.value, denom)
    prob_sum = (parts.upup.value + parts.updown.value + parts.downup.value + parts.downdown.value) / denom.value
    @printf("%-24s sumP  = %.16g\n", "projector BP", prob_sum)
    @printf("%-24s <ZZ>  = %.16g\n", "projector BP", zz)
    return zz
end

function run_loop_zz(psi, cubic; maxiter::Int64, tol::Float64, damping::Float64)
    println("loop correction uses pair-normalized BP messages internally")
    zz, parts, denom = loop_sink_zz_projector(psi, cubic; maxiter=maxiter, tol=tol, damping=damping)
    print_loop_result("loop projector up-up", parts.upup)
    print_loop_result("loop projector up-down", parts.updown)
    print_loop_result("loop projector down-up", parts.downup)
    print_loop_result("loop projector down-down", parts.downdown)
    print_loop_result("loop denominator", denom)
    prob_sum = (parts.upup.value + parts.updown.value + parts.downup.value + parts.downdown.value) / denom.value
    @printf("%-24s sumP  = %.16g\n", "loop projector BP", prob_sum)
    @printf("%-24s <ZZ>  = %.16g\n", "loop projector BP", zz)
    return zz
end

function main(args)
    opts = parse_bp_args(args)
    lx = parse(Int, opts["lx"])
    ly = parse(Int, opts["ly"])
    lz = parse(Int, opts["lz"])
    theta = parse(Float64, opts["theta"])
    chi = parse(Int, opts["chi"])
    cutoff = parse(Float64, opts["cutoff"])
    maxiter = parse(Int, opts["maxiter"])
    tol = parse(Float64, opts["tol"])
    damping = parse(Float64, opts["damping"])
    pair_normalize = parse_bool(opts["pairnorm"])
    method = opts["method"]

    @printf("Preparing cubic whip TNS: L = (%d,%d,%d), theta = %.16g\n", lx, ly, lz, theta)
    psi, cubic, _ = prepare_cubic_whip_tns(lx, ly, lz, theta; chi=chi, cutoff=cutoff)
    i, j = last_sink_bond(cubic)
    println("sink bond: $i $(idx2coord(i, cubic.dims)) -> $j $(idx2coord(j, cubic.dims))")
    @printf("BP params: maxiter = %d, tol = %.3e, damping = %.3f, pairnorm = %s\n\n", maxiter, tol, damping, string(pair_normalize))

    if method == "direct" || method == "all"
        run_direct_zz(psi, cubic; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=pair_normalize)
        println()
    end
    if method == "projector" || method == "all"
        run_projector_zz(psi, cubic; maxiter=maxiter, tol=tol, damping=damping, pair_normalize=pair_normalize)
        println()
    end
    if method == "loop" || method == "with_loop" || method == "all"
        run_loop_zz(psi, cubic; maxiter=maxiter, tol=tol, damping=damping)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
