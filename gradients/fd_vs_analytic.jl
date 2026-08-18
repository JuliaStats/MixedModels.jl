# Compare gradient-based optimization using the analytic gradient
# (objective_gradient!) against ForwardDiff-based gradients, on a
# many-parameter model (kb07 maximal) and a tall model (ml1m, ~1M rows).
#
# Usage (driver mode, spawns one subprocess per configuration):
#
#     julia --startup-file=no --project=gradients gradients/fd_vs_analytic.jl
#
# Each configuration runs in a fresh subprocess because Sys.maxrss() is
# monotone within a process; the peak-RSS column is only meaningful per process.
# Within each worker the first fit warms up compilation and the measurement is
# taken on a refit!, so time/bytes/allocs exclude compilation.

using MixedModels

if length(ARGS) == 3 && ARGS[3] == "forwarddiff"
    using ForwardDiff
end

const FORMS = Dict(
    "kb07" => (
        @formula(
            rt_trunc ~
                1 + spkr * prec * load +
                (1 + spkr + prec + load | subj) +
                (1 + spkr + prec + load | item)
        ),
        Dict{Symbol,Any}(),
    ),
    "ml1m" => (
        @formula(Y ~ 1 + (1 | G) + (1 | H)),
        Dict{Symbol,Any}(:G => Grouping(), :H => Grouping()),
    ),
)

function runconfig(dsname::String, optimizer::String, gradient::String)
    form, contrasts = FORMS[dsname]
    tbl = MixedModels.dataset(Symbol(dsname))
    # warm up: compiles the full fit path for this configuration
    m = fit(MixedModel, form, tbl;
        contrasts, progress=false,
        optimizer=Symbol(optimizer), gradient=Symbol(gradient))
    GC.gc()
    stats = @timed refit!(m; progress=false)
    println(
        join(
            [
                "RESULT", dsname, optimizer, gradient,
                length(m.parmap), m.optsum.feval, m.optsum.fmin,
                stats.time, stats.bytes, Base.gc_alloc_count(stats.gcstats),
                Sys.maxrss(),
            ],
            "\t"),
    )
    return nothing
end

const CONFIGS = [
    (ds, opt, grad) for ds in ("kb07", "ml1m") for
    (opt, grad) in
    (("LN_NEWUOA", "analytic"), ("LD_LBFGS", "analytic"), ("LD_LBFGS", "forwarddiff"))
]

mib(bytes) = string(round(bytes / 2^20; digits=1))

function driver()
    rows = Vector{Vector{String}}()
    for (ds, opt, grad) in CONFIGS
        @info "running" ds opt grad
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $(@__FILE__) $ds $opt $grad`
        out = read(cmd, String)
        line = only(filter(startswith("RESULT"), split(out, '\n')))
        push!(rows, string.(split(line, '\t')[2:end]))
    end
    println()
    println(
        "| model | optimizer | gradient | nθ | feval | objective | time (s) | alloc (MiB) | # allocs | peak RSS (MiB) |",
    )
    println("|---|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows
        ds, opt, grad, ntheta, feval, fmin, time, bytes, allocs, maxrss = r
        gradlabel = startswith(opt, "LD") ? grad : "—"
        println("| ",
            join(
                [
                    ds, opt, gradlabel, ntheta, feval,
                    string(round(parse(Float64, fmin); digits=4)),
                    string(round(parse(Float64, time); digits=3)),
                    mib(parse(Int, bytes)),
                    allocs,
                    mib(parse(Int, maxrss)),
                ],
                " | "), " |")
    end
    return nothing
end

if length(ARGS) == 3
    runconfig(ARGS...)
else
    driver()
end
