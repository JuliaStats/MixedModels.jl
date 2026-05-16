using JSON3

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        i == length(args) && error("Missing value for option $arg")
        opts[arg] = args[i + 1]
        i += 2
    end
    return opts
end

function usage()
    return """
    Usage:
      julia --project=. benchmark/compare_fitted_predict.jl \\
        --lhs PATH \\
        --rhs PATH \\
        --dataset DATASET \\
        --formula 'response ~ ...' \\
        [--group GROUPING_COLUMN] \\
        [--family FAMILY] \\
        [--link LINK] \\
        [--iterations N] \\
        [--format text|json|markdown] \\
        [--lhs-label LABEL] \\
        [--rhs-label LABEL]
    """
end

function run_script(dir::String, script_args::Vector{String})
    script = joinpath(@__DIR__, "fitted_predict.jl")
    cmd = Cmd([Base.julia_cmd().exec..., "--project=$dir", script, script_args...])
    io = IOBuffer()
    run(pipeline(cmd; stdout=io))
    return JSON3.read(String(take!(io)))
end

function metric_names(result)
    return filter(name -> getproperty(result, Symbol(name)) !== nothing,
        ("fitted_inplace", "fitted", "predict_same", "predict_population_new_level"))
end

function pct_change(old, new)
    return old == 0 ? missing : (new - old) / old
end

function compare(lhs, rhs)
    metrics = metric_names(lhs)
    return map(metrics) do name
        l = getproperty(lhs, Symbol(name))
        r = getproperty(rhs, Symbol(name))
        (
            benchmark=name,
            lhs_alloc=l.alloc,
            rhs_alloc=r.alloc,
            alloc_delta=r.alloc - l.alloc,
            alloc_pct=pct_change(l.alloc, r.alloc),
            lhs_time=l.time,
            rhs_time=r.time,
            time_delta=r.time - l.time,
            time_pct=pct_change(l.time, r.time),
        )
    end
end

function render_text(rows, lhs_label, rhs_label)
    println(rpad("benchmark", 30), lpad("$lhs_label alloc", 14),
        lpad("$rhs_label alloc", 14),
        lpad("alloc %", 12), lpad("$lhs_label time", 16), lpad("$rhs_label time", 16),
        lpad("time %", 12))
    for row in rows
        println(
            rpad(row.benchmark, 30),
            lpad(string(row.lhs_alloc), 14),
            lpad(string(row.rhs_alloc), 14),
            lpad(
                if isnothing(row.alloc_pct) || ismissing(row.alloc_pct)
                    "missing"
                else
                    string(round(100 * row.alloc_pct; digits=2), "%")
                end,
                12,
            ),
            lpad(string(row.lhs_time), 16),
            lpad(string(row.rhs_time), 16),
            lpad(
                if isnothing(row.time_pct) || ismissing(row.time_pct)
                    "missing"
                else
                    string(round(100 * row.time_pct; digits=2), "%")
                end,
                12,
            ),
        )
    end
end

function render_markdown(rows, lhs_label, rhs_label)
    println(
        "| benchmark | $lhs_label alloc | $rhs_label alloc | alloc % | $lhs_label time | $rhs_label time | time % |",
    )
    println("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows
        alloc_pct = if isnothing(row.alloc_pct) || ismissing(row.alloc_pct)
            "missing"
        else
            string(round(100 * row.alloc_pct; digits=2), "%")
        end
        time_pct = if isnothing(row.time_pct) || ismissing(row.time_pct)
            "missing"
        else
            string(round(100 * row.time_pct; digits=2), "%")
        end
        println("| ", row.benchmark, " | ", row.lhs_alloc, " | ", row.rhs_alloc, " | ",
            alloc_pct, " | ", row.lhs_time, " | ", row.rhs_time, " | ", time_pct, " |")
    end
end

function main(args)
    isempty(args) && error(usage())
    opts = parse_args(args)
    for required in ("--lhs", "--rhs", "--dataset", "--formula")
        haskey(opts, required) || error("Missing required option $required\n\n$(usage())")
    end

    lhs = opts["--lhs"]
    rhs = opts["--rhs"]
    lhs_label = get(opts, "--lhs-label", "lhs")
    rhs_label = get(opts, "--rhs-label", "rhs")
    format = get(opts, "--format", "text")

    script_args = String[
        "--dataset", opts["--dataset"],
        "--formula", opts["--formula"],
        "--iterations", get(opts, "--iterations", "30"),
        "--format", "json",
    ]
    for opt in ("--group", "--family", "--link")
        haskey(opts, opt) && append!(script_args, [opt, opts[opt]])
    end

    lhs_result = run_script(lhs, script_args)
    rhs_result = run_script(rhs, script_args)
    rows = compare(lhs_result, rhs_result)

    if format == "text"
        render_text(rows, lhs_label, rhs_label)
    elseif format == "json"
        JSON3.write(stdout, (; lhs=lhs_result, rhs=rhs_result, comparison=rows))
        println()
    elseif format == "markdown"
        render_markdown(rows, lhs_label, rhs_label)
    else
        error("Unsupported format: $format")
    end
end

main(ARGS)
