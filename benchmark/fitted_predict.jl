using JSON3
using MixedModels
using Tables

const BENCHMARK_CONTRASTS = Dict{Symbol,Any}(
    :batch => Grouping(),
    :cask => Grouping(),
    :d => Grouping(),
    :g => Grouping(),
    :h => Grouping(),
    :i => Grouping(),
    :item => Grouping(),
    :Machine => Grouping(),
    :plate => Grouping(),
    :s => Grouping(),
    :sample => Grouping(),
    :subj => Grouping(),
    :Worker => Grouping(),
    :F => HelmertCoding(),
    :P => HelmertCoding(),
    :Q => HelmertCoding(),
    :lQ => HelmertCoding(),
    :lT => HelmertCoding(),
    :load => HelmertCoding(),
    :prec => HelmertCoding(),
    :service => HelmertCoding(),
    :spkr => HelmertCoding(),
)

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
      julia --project=. benchmark/fitted_predict.jl \\
        --dataset DATASET \\
        --formula 'response ~ ...' \\
        [--group GROUPING_COLUMN] \\
        [--family FAMILY] \\
        [--link LINK] \\
        [--iterations N] \\
        [--format text|json|markdown]

    Options:
      --dataset     Dataset name understood by MixedModels.dataset, e.g. sleepstudy
      --formula     Formula contents without the @formula wrapper
      --group       Optional grouping column to perturb for the new-level prediction benchmark
      --family      Optional distribution, e.g. Bernoulli or Poisson()
      --link        Optional link, e.g. LogitLink or ProbitLink()
      --iterations  Number of timed iterations per benchmark (default: 30)
      --format      Output format: text, json, or markdown (default: text)
    """
end

parse_formula(str::AbstractString) = Base.eval(Main, Meta.parse("@formula($str)"))

function parse_entity(str::AbstractString)
    val = Base.eval(Main, Meta.parse(str))
    return val isa Type ? val() : val
end

function bench(f, n::Integer)
    GC.gc()
    alloc = @allocated f()
    GC.gc()
    t = @elapsed for _ in 1:n
        f()
    end
    return (; alloc, time=t / n)
end

function perturb_levels(tbl::NamedTuple, group::Symbol)
    vals = Tables.getcolumn(tbl, group)
    isempty(vals) && error("Cannot perturb empty grouping column $group")
    newvals = map(x -> x == first(vals) ? "NEW" : x, vals)
    return merge(tbl, (; group => newvals))
end

function fit_model(formula, tbl, family, link)
    kwargs = (; contrasts=BENCHMARK_CONTRASTS, progress=false)
    if isnothing(family)
        return fit(MixedModel, formula, tbl; kwargs...)
    elseif isnothing(link)
        return fit(MixedModel, formula, tbl, family; kwargs...)
    else
        return fit(MixedModel, formula, tbl, family, link; kwargs...)
    end
end

function run_benchmarks(
    dataset_name::Symbol,
    formula;
    group::Union{Nothing,Symbol},
    family=nothing,
    link=nothing,
    iterations::Int,
)
    tbl = Tables.columntable(MixedModels.dataset(dataset_name))
    model = fit_model(formula, tbl, family, link)
    fitted_buf = similar(fitted(model))

    same_data = tbl
    new_level_data = isnothing(group) ? nothing : perturb_levels(tbl, group)

    fitted!(fitted_buf, model)
    fitted(model)
    predict(model, same_data)
    isnothing(new_level_data) || predict(model, new_level_data; new_re_levels=:population)

    results = (
        dataset=String(dataset_name),
        formula=string(formula),
        family=isnothing(family) ? nothing : string(typeof(family)),
        link=isnothing(link) ? nothing : string(typeof(link)),
        group=isnothing(group) ? nothing : String(group),
        iterations=iterations,
        fitted_inplace=bench(() -> fitted!(fitted_buf, model), iterations),
        fitted=bench(() -> fitted(model), iterations),
        predict_same=bench(() -> predict(model, same_data), iterations),
        predict_population_new_level=if isnothing(group)
            nothing
        else
            bench(
            () -> predict(model, new_level_data; new_re_levels=:population), iterations
        )
        end,
    )

    return results
end

function render_text(result)
    println("dataset: ", result.dataset)
    println("formula: ", result.formula)
    isnothing(result.family) || println("family: ", result.family)
    isnothing(result.link) || println("link: ", result.link)
    isnothing(result.group) || println("group: ", result.group)
    println("iterations: ", result.iterations)
    println()
    println(rpad("benchmark", 30), lpad("alloc (B)", 14), lpad("time (s)", 16))
    for name in (:fitted_inplace, :fitted, :predict_same, :predict_population_new_level)
        val = getproperty(result, name)
        isnothing(val) && continue
        println(
            rpad(String(name), 30), lpad(string(val.alloc), 14), lpad(string(val.time), 16)
        )
    end
end

function render_markdown(result)
    println("| benchmark | alloc (B) | time (s) |")
    println("|---|---:|---:|")
    for name in (:fitted_inplace, :fitted, :predict_same, :predict_population_new_level)
        val = getproperty(result, name)
        isnothing(val) && continue
        println("| ", name, " | ", val.alloc, " | ", val.time, " |")
    end
end

function main(args)
    isempty(args) && error(usage())
    opts = parse_args(args)
    haskey(opts, "--dataset") || error("Missing required option --dataset\n\n$(usage())")
    haskey(opts, "--formula") || error("Missing required option --formula\n\n$(usage())")

    dataset_name = Symbol(opts["--dataset"])
    formula = parse_formula(opts["--formula"])
    group = haskey(opts, "--group") ? Symbol(opts["--group"]) : nothing
    family = haskey(opts, "--family") ? parse_entity(opts["--family"]) : nothing
    link = haskey(opts, "--link") ? parse_entity(opts["--link"]) : nothing
    iterations = parse(Int, get(opts, "--iterations", "30"))
    format = get(opts, "--format", "text")

    result = run_benchmarks(dataset_name, formula; group, family, link, iterations)

    if format == "text"
        render_text(result)
    elseif format == "json"
        JSON3.write(stdout, result)
        println()
    elseif format == "markdown"
        render_markdown(result)
    else
        error("Unsupported format: $format")
    end
end

main(ARGS)
