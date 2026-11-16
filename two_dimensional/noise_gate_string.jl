using Random

function main()
    # Define the parameters for the noise gate string
    L = parse(Int, ARGS[1])
    NSAMPLE = parse(Int, ARGS[2])
    SEED = parse(Int, ARGS[3])
    Random.seed!(SEED)
    noise_gate_string = ""
    for _ in 1:NSAMPLE
        # Generate a random noise gate string based on the parameters
        b1 = rand(1:L)
        b2 = rand(1:L)
        d = rand(["+x","+y"])
        if d == "+x"
            noise_gate = "$b1,$b2>$(b1+1),$b2;"
        elseif d == "+y"
            noise_gate = "$b1,$b2>$b1,$(b2+1);"
        end
        if (b1 == L)
            noise_gate = "$b1,$b2>$b1,$(b2+1);"
        elseif (b2 == L)
            noise_gate = "$b1,$b2>$(b1+1),$b2;"
        end

        noise_gate_string *= noise_gate
    end
    # Output the generated noise gate string
    println("Sample: $noise_gate_string")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end