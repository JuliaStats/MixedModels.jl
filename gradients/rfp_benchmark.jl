# Benchmark the four combinations (gradient LD_LBFGS / derivative-free LN_NEWUOA)
# × (RFP / no RFP) on the insteval and ml1m datasets, both of which cross the
# default RFPthreshold (insteval d-block: 1128 columns, ml1m movie block: 3706).
#
# Usage (driver mode, spawns one subprocess per configuration):
#
#     julia --startup-file=no --project=. gradients/rfp_benchmark.jl
#
# Each configuration runs in a fresh subprocess because Sys.maxrss() is
# monotone within a process; the peak-RSS column is only meaningful per process.
# Within each worker the first fit warms up compilation and the measurement is
# taken on a refit!, so time/bytes exclude compilation (peak RSS does not).

using MixedModels
using MixedModels: GradientWorkspace, TriangularRFP

const FORMS = Dict(
    "insteval" => (
        @formula(y ~ 1 + service + (1 | s) + (1 | d)),
        Dict{Symbol,Any}(:s => Grouping(), :d => Grouping()),
    ),
    "ml1m" => (
        @formula(Y ~ 1 + (1 | G) + (1 | H)),
        Dict{Symbol,Any}(:G => Grouping(), :H => Grouping()),
    ),
)

function runconfig(dsname::String, optimizer::String, rfp::String)
    form, contrasts = FORMS[dsname]
    tbl = MixedModels.dataset(Symbol(dsname))
    RFPthreshold = rfp == "rfp" ? 1000 : typemax(Int)
    # warm up: compiles the full fit path for this configuration
    m = fit(MixedModel, form, tbl;
        contrasts, progress=false, RFPthreshold,
        optimizer=Symbol(optimizer))
    nrfp = count(Base.Fix2(isa, TriangularRFP), m.L)
    @assert (rfp == "rfp") == (nrfp > 0)
    GC.gc()
    stats = @timed refit!(m; progress=false)
    wsbytes = startswith(optimizer, "LD") ? Base.summarysize(GradientWorkspace(m)) : 0
    println(
        join(
            [
                "RESULT", dsname, optimizer, rfp,
                m.optsum.feval, m.optsum.fmin,
                stats.time,
                Base.summarysize(m), Base.summarysize(m.L), wsbytes,
                Sys.maxrss(),
            ],
            "\t"),
    )
    return nothing
end

const CONFIGS = [
    (ds, opt, rfp) for ds in ("insteval", "ml1m") for
    opt in ("LN_NEWUOA", "LD_LBFGS") for rfp in ("dense", "rfp")
]

mib(bytes) = string(round(bytes / 2^20; digits=1))

function driver()
    rows = Vector{Vector{String}}()
    for (ds, opt, rfp) in CONFIGS
        @info "running" ds opt rfp
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(@__DIR__)) $(@__FILE__) $ds $opt $rfp`
        out = read(cmd, String)
        line = only(filter(startswith("RESULT"), split(out, '\n')))
        push!(rows, string.(split(line, '\t')[2:end]))
    end
    println()
    println(
        "| model | optimizer | storage | model (MiB) | L (MiB) | grad ws (MiB) | peak RSS (MiB) | time (s) | feval | ms/feval | objective |",
    )
    println("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows
        ds, opt, rfp, feval, fmin, time, msize, lsize, wsize, maxrss = r
        t = parse(Float64, time)
        nfeval = parse(Int, feval)
        println("| ",
            join(
                [
                    ds, opt, rfp,
                    mib(parse(Int, msize)),
                    mib(parse(Int, lsize)),
                    iszero(parse(Int, wsize)) ? "—" : mib(parse(Int, wsize)),
                    mib(parse(Int, maxrss)),
                    string(round(t; digits=2)),
                    feval,
                    string(round(1000 * t / nfeval; digits=1)),
                    string(round(parse(Float64, fmin); digits=4)),
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
