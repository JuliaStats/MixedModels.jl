# Comprehensive comparison of the optimizers available for a `LinearMixedModel`:
# the derivative-free `LN_NEWUOA` and `LN_BOBYQA` against the gradient-based
# `LD_LBFGS` (and optionally `LD_MMA` / `LD_SLSQP`), driven either by the analytic
# gradient (`objective_gradient!`) or by forward-mode automatic differentiation.
#
# The model suite spans data sizes from 60 to 2.7×10^6 observations, parameter
# counts from 1 to 36, and deliberately different block structures for `L`:
# single grouping factors, nested factors (sparse off-diagonal blocks), crossed
# factors (dense fill-in), scalar terms (`Diagonal` diagonal blocks),
# vector-valued terms (`UniformBlockDiagonal`), `zerocorr` (diagonal `Λ`), and
# both few-and-many fixed-effects columns.  Run with `--shapes` to print the
# `BlockDescription` of every model in the suite.
#
# Usage (driver mode; spawns one subprocess per configuration):
#
#     julia --startup-file=no --project=gradients gradients/optimizer_benchmark.jl [options]
#
# Options are of the form `--key=value`; `--key` means `--key=true` and `--no-key`
# means `--key=false`:
#
#     --tiers=small,medium,large   size tiers to run (`huge` is opt-in)
#     --models=kb07,d3             explicit model list, overrides --tiers
#     --optimizers=LN_NEWUOA,...   optimizers to fit with
#     --gradients=analytic,forwarddiff   gradient sources for the `LD_*` optimizers
#     --reps=1                     timed refits per configuration (the fastest is reported)
#     --seconds=5                  per-kernel time budget for the per-evaluation table
#     --maxtime=-1                 `OptSummary.maxtime` in seconds; a positive value caps
#                                  each fit, which shows up as `MAXTIME_REACHED` in the
#                                  return column and makes that row's time a lower bound
#     --rfpthreshold=1000          `RFPthreshold`; a huge value forces dense diagonal blocks
#     --no-evals                   skip the per-evaluation table
#     --no-fits                    skip the fit table
#     --shapes                     also print the full `BlockDescription` of each model
#     --list                       list the model suite and exit
#
# Every configuration runs in a fresh subprocess: `Sys.maxrss()` is monotone
# within a process, so the peak-RSS column is only meaningful per process, and
# compilation of a code path should not be charged to whichever configuration
# happens to reach it first.  Within a worker the first fit warms up compilation
# and the measurement is taken on a `refit!`, so time and allocations exclude
# compilation (peak RSS does not).
#
# The whole default suite takes on the order of half an hour, nearly all of it in
# `ml1m` with ForwardDiff.  `--tiers=small` runs in a couple of minutes.

using MixedModels
using MixedModels: FormulaTerm, GradientWorkspace
using MixedModelsDatasets: dataset
using LinearAlgebra: BLAS
using Printf: @sprintf

#####
##### the model suite
#####

struct BenchModel
    name::String
    tier::Symbol
    dsname::Symbol
    formula::FormulaTerm
    contrasts::Dict{Symbol,Any}
    # whether to run the ForwardDiff configurations for this model.  It is off for
    # the `huge` tier, where a dual-valued sweep over the whole response is
    # expensive rather than impossible; flip it to run those too.
    fd::Bool
    note::String  # the shape this model contributes to the suite
end

grp(syms::Symbol...) = Dict{Symbol,Any}(s => Grouping() for s in syms)
helm(syms::Symbol...) = Dict{Symbol,Any}(s => HelmertCoding() for s in syms)

const MODELS = BenchModel[
    BenchModel("sleepstudy1", :small, :sleepstudy,
        @formula(reaction ~ 1 + days + (1 | subj)),
        grp(:subj), true,
        "one scalar term, a single parameter (NEWUOA falls back to BOBYQA)"),
    BenchModel("sleepstudy_zc", :small, :sleepstudy,
        @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
        grp(:subj), true,
        "one vector-valued term with diagonal Λ"),
    BenchModel("sleepstudy", :small, :sleepstudy,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        grp(:subj), true,
        "one vector-valued term, 2×2 faces"),
    BenchModel("pastes", :small, :pastes,
        @formula(strength ~ 1 + (1 | batch / cask)),
        grp(:batch, :cask), true,
        "nested scalar terms, sparse off-diagonal block"),
    BenchModel("penicillin", :small, :penicillin,
        @formula(diameter ~ 1 + (1 | plate) + (1 | sample)),
        grp(:plate, :sample), true,
        "small fully crossed scalar terms"),
    BenchModel("oxide", :small, :oxide,
        @formula(Thickness ~ 1 + Source + (1 + Source | Lot) + (1 + Source | Lot & Wafer)),
        Dict{Symbol,Any}(), true,
        "nested vector-valued terms, UniformBlockDiagonal diagonal blocks"),
    BenchModel("kb07", :small, :kb07,
        @formula(rt_trunc ~
            1 + spkr * prec * load +
            (1 + spkr + prec + load | subj) + (1 + spkr + prec + load | item)),
        merge(grp(:subj, :item), helm(:spkr, :prec, :load)), true,
        "maximal crossed vector-valued model: 20θ on only 1789 rows"),
    BenchModel("kwdyz11", :medium, :kwdyz11,
        @formula(rt ~ 1 + CTR + (1 + CTR | Subj) + (1 + CTR | Item)),
        grp(:Subj, :Item), true,
        "crossed 4-column vector-valued terms, n = 28710"),
    BenchModel("kkl15", :medium, :kkl15,
        @formula(rt ~ 1 + CTR + (1 + CTR | Subj)),
        grp(:Subj), true,
        "a single 4-column vector-valued term, n = 53765"),
    BenchModel("mrk17", :medium, :mrk17_exp1,
        @formula(1000 / rt ~
            1 + F * P * Q * lQ * lT +
            (1 + P + Q + lQ + lT | item) + (1 + F + P + Q + lQ + lT | subj)),
        merge(grp(:subj, :item), helm(:F, :P, :Q, :lQ, :lT)), true,
        "the most parameters in the suite: 36θ and p = 32"),
    BenchModel("insteval", :medium, :insteval,
        @formula(y ~ 1 + service + (1 | s) + (1 | d) + (1 | dept)),
        merge(grp(:s, :d), helm(:service)), true,
        "three crossed scalar terms, n = 73421, sparse cross blocks"),
    BenchModel("insteval_fe", :medium, :insteval,
        @formula(y ~ 1 + service * dept + (1 | s) + (1 | d)),
        merge(grp(:s, :d), helm(:service)), true,
        "two crossed scalar terms but p = 28 fixed-effects columns"),
    BenchModel("insteval_vec", :medium, :insteval,
        @formula(y ~ 1 + service + (1 | s) + (1 | d) + (1 + service | dept)),
        merge(grp(:s, :d), helm(:service)), true,
        "scalar and vector-valued terms mixed in one model"),
    BenchModel("d3", :medium, :d3,
        @formula(y ~ 1 + u + (1 + u | g) + (1 + u | h) + (1 + u | i)),
        grp(:g, :h, :i), true,
        "three crossed vector-valued terms, n = 130418"),
    BenchModel("ml1m", :large, :ml1m,
        @formula(Y ~ 1 + (1 | G) + (1 | H)),
        grp(:G, :H), true,
        "n = 10^6 with only 2θ; the fill-in lands in a 3706×3706 diagonal block, " *
        "which is stored in RFP format at the default threshold"),
    BenchModel("elp_ldt", :huge, :elp_ldt_trial,
        @formula(rt ~ 1 + (1 | subj) + (1 | item)),
        grp(:subj, :item), false,
        "n = 2745952, 80962 items crossed with 814 subjects; a structurally small " *
        "`L` whose cost is entirely in the sweep over the observations"),
]

modelnames() = [bm.name for bm in MODELS]

function getbenchmodel(name::AbstractString)
    idx = findfirst(bm -> bm.name == name, MODELS)
    isnothing(idx) &&
        throw(ArgumentError("unknown model $name; known models are $(modelnames())"))
    return MODELS[idx]
end

"""
    buildmodel(bm::BenchModel, rfpthreshold::Int)

Construct, but do not fit, the `LinearMixedModel` described by `bm`.
"""
function buildmodel(bm::BenchModel, rfpthreshold::Int)
    return LinearMixedModel(bm.formula, dataset(bm.dsname);
        contrasts=bm.contrasts, RFPthreshold=rfpthreshold)
end

#####
##### worker/driver protocol
#####

# A worker reports each measurement as one line: a tag followed by tab-separated
# `key=value` pairs, so that adding a field does not disturb the parsing.
function report(tag::AbstractString; kwargs...)
    println(join([tag, ("$k=$v" for (k, v) in kwargs)...], '\t'))
    return nothing
end

function parseresult(line::AbstractString)
    return Dict{String,String}(
        String(first(kv)) => String(last(kv))
        for kv in (split(f, '='; limit=2) for f in split(line, '\t')[2:end])
    )
end

num(d::Dict{String,String}, k) = parse(Float64, d[k])
int(d::Dict{String,String}, k) = parse(Int, d[k])

#####
##### worker: block structure and model metadata
#####

function worker_shape(name::AbstractString, rfpthreshold::Int)
    bm = getbenchmodel(name)
    m = buildmodel(bm, rfpthreshold)
    bd = BlockDescription(m)
    k = length(bd.blknms) - 1   # the compact form omits the fixed-effects row
    report("SHAPE";
        model=name, tier=bm.tier, dataset=bm.dsname,
        n=nobs(m), p=size(m.X, 2), ntheta=length(m.θ), nre=length(m.reterms),
        qs=join(bd.blkrows[1:k], ","),
        shape=join((join(bd.ALtypes[i, 1:i], ",") for i in 1:k), " ; "),
        Lbytes=Base.summarysize(m.L))
    println("BLOCKS")
    show(stdout, MIME"text/plain"(), bd)
    println("ENDBLOCKS")
    return nothing
end

#####
##### worker: the cost of a single objective or gradient evaluation
#####

"""
    evalbench(f, seconds)

Call `f` repeatedly, for up to `seconds` or `maxsamples` calls, and return the
fastest observed time together with the allocations of a single call.  The
minimum is the appropriate summary here: these kernels are deterministic, so
anything above the minimum is interference from the rest of the machine.
"""
function evalbench(f, seconds::Float64; maxsamples::Int=10_000)
    f()                  # warm up: the first call compiles, and allocates while doing so
    stats = @timed f()   # the allocation figures come from a single steady-state call
    elapsed = 0.0
    best = Inf
    samples = 0
    while elapsed < seconds && samples < maxsamples
        s = @timed f()
        best = min(best, s.time)
        elapsed += s.time
        samples += 1
    end
    return (time=best, alloc=stats.bytes, allocs=Base.gc_alloc_count(stats.gcstats))
end

function worker_eval(name::AbstractString, rfpthreshold::Int, seconds::Float64)
    bm = getbenchmodel(name)
    m = buildmodel(bm, rfpthreshold)
    # measure at the optimum: that is where an optimizer spends most of its
    # evaluations, and the fill-in pattern there is the one that matters
    fit!(m; progress=false)
    θ = copy(m.θ)
    g = similar(θ)
    ws = GradientWorkspace(m)
    obj = evalbench(() -> objective!(m, θ), seconds)
    grad = evalbench(() -> objective_gradient!(ws, g, m, θ), seconds)
    objective_gradient!(ws, g, m, θ)

    fdtime = fdalloc = fdallocs = fdwsbytes = -1
    reldiff = NaN
    if bm.fd
        gfd = similar(θ)
        fdws = MixedModels.fd_gradient_workspace(m)
        MixedModels.fd_objective_gradient!(fdws, gfd, m, θ)
        reldiff = maximum(abs, g - gfd) / max(1.0, maximum(abs, g))
        fd = evalbench(() -> MixedModels.fd_objective_gradient!(fdws, gfd, m, θ), seconds)
        fdtime, fdalloc, fdallocs = fd.time, fd.alloc, fd.allocs
        fdwsbytes = Base.summarysize(fdws)
    end

    report("EVAL";
        model=name, ntheta=length(θ),
        objtime=obj.time, objalloc=obj.alloc,
        gradtime=grad.time, gradalloc=grad.alloc, gradallocs=grad.allocs,
        fdtime, fdalloc, fdallocs,
        wsbytes=Base.summarysize(ws), fdwsbytes,
        reldiff, maxg=maximum(abs, g))
    return nothing
end

#####
##### worker: a complete fit
#####

function worker_fit(name::AbstractString, optimizer::AbstractString,
    gradient::AbstractString, rfpthreshold::Int, reps::Int, maxtime::Float64)
    bm = getbenchmodel(name)
    m = buildmodel(bm, rfpthreshold)
    opt, grad = Symbol(optimizer), Symbol(gradient)
    m.optsum.maxtime = maxtime  # negative means no limit; applies to every fit below
    # warm up: compiles the full fit path for this configuration
    fit!(m; progress=false, optimizer=opt, gradient=grad)
    best = (time=Inf, bytes=0, allocs=0)
    for _ in 1:reps
        GC.gc()
        stats = @timed refit!(m; progress=false)
        if stats.time < best.time
            best = (time=stats.time, bytes=stats.bytes,
                allocs=Base.gc_alloc_count(stats.gcstats))
        end
    end
    # capture RSS before allocating anything for the diagnostics below, so that a
    # derivative-free row is not charged for a gradient workspace it never needed
    maxrss = Sys.maxrss()

    # the gradient at the returned optimum measures how tightly each optimizer
    # actually converged, on a scale that is comparable across configurations
    maxg = try
        g = similar(m.θ)
        objective_gradient!(g, m, copy(m.θ))
        maximum(abs, g)
    catch err
        @warn "gradient diagnostic failed" name optimizer exception = err
        NaN
    end

    report("FIT";
        model=name, optimizer, gradient, effopt=m.optsum.optimizer,
        ntheta=length(m.θ), feval=m.optsum.feval, fmin=m.optsum.fmin,
        time=best.time, bytes=best.bytes, allocs=best.allocs, maxrss,
        maxg, singular=issingular(m), ret=m.optsum.returnvalue)
    return nothing
end

#####
##### driver: option handling
#####

const DEFAULTS = Dict{String,String}(
    "tiers" => "small,medium,large",
    "models" => "",
    "optimizers" => "LN_NEWUOA,LN_BOBYQA,LD_LBFGS",
    "gradients" => "analytic,forwarddiff",
    "reps" => "1",
    "seconds" => "5",
    "maxtime" => "-1",
    "rfpthreshold" => "1000",
    "evals" => "true",
    "fits" => "true",
    "shapes" => "false",
    "list" => "false",
)

function parseoptions(args)
    opts = copy(DEFAULTS)
    for a in args
        startswith(a, "--") ||
            throw(ArgumentError("expected an option starting with `--`, got $a"))
        body = a[3:end]
        i = findfirst(==('='), body)
        key, val = if isnothing(i)   # --flag and --no-flag
            startswith(body, "no-") ? (body[4:end], "false") : (body, "true")
        else
            (body[1:prevind(body, i)], body[nextind(body, i):end])
        end
        haskey(opts, key) || throw(
            ArgumentError("unknown option --$key; known options are $(sort!(collect(keys(opts))))"))
        opts[key] = val
    end
    return opts
end

split_list(s::AbstractString) = String.(filter!(!isempty, strip.(split(s, ','))))
flag(opts, key) = parse(Bool, opts[key])

function selectmodels(opts)
    names = split_list(opts["models"])
    isempty(names) || return [getbenchmodel(n) for n in names]
    tiers = Symbol.(split_list(opts["tiers"]))
    return filter(bm -> bm.tier in tiers, MODELS)
end

"""
    configs(models, opts)

The product of models and optimizers, expanded over gradient sources for the
gradient-based (`LD_*`) optimizers.  Models flagged as impractical for
ForwardDiff are run with the analytic gradient only.
"""
function configs(models, opts)
    grads = split_list(opts["gradients"])
    out = Tuple{BenchModel,String,String}[]
    for bm in models, o in split_list(opts["optimizers"])
        if startswith(o, "LD")
            for g in grads
                (g == "forwarddiff" && !bm.fd) && continue
                push!(out, (bm, o, g))
            end
        else
            push!(out, (bm, o, "analytic"))  # ignored by a derivative-free optimizer
        end
    end
    return out
end

function runworker(args::Vector{String})
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $(@__FILE__) $args`
    return read(cmd, String)
end

function collectlines(out::AbstractString, tag::AbstractString)
    return [parseresult(l) for l in split(out, '\n') if startswith(l, tag * '\t')]
end

function extractblocks(out::AbstractString)
    lines = split(out, '\n')
    i = findfirst(==("BLOCKS"), lines)
    j = findfirst(==("ENDBLOCKS"), lines)
    return (isnothing(i) || isnothing(j)) ? "" : join(lines[(i + 1):(j - 1)], '\n')
end

#####
##### driver: formatting
#####

mib(bytes) = bytes < 0 ? "—" : @sprintf("%.1f", bytes / 2^20)
kib(bytes) = bytes < 0 ? "—" : @sprintf("%.1f", bytes / 2^10)
ms(seconds) = seconds < 0 ? "—" : @sprintf("%.3f", 1000 * seconds)
sec(seconds) = @sprintf("%.3f", seconds)
ratio(a, b) = (a < 0 || b <= 0) ? "—" : @sprintf("%.1f×", a / b)

function mdtable(headers, aligns, rows)
    println("| ", join(headers, " | "), " |")
    println("|", join((a === :r ? "--:" : "---" for a in aligns), "|"), "|")
    for r in rows
        println("| ", join(r, " | "), " |")
    end
    println()
    return nothing
end

function header(opts, models)
    println("# Optimizer benchmark for `LinearMixedModel`\n")
    println("* julia ", VERSION, " on ", Sys.CPU_NAME, " (", Sys.CPU_THREADS, " threads)")
    println("* BLAS: ", basename(first(BLAS.get_config().loaded_libs).libname), " with ",
        BLAS.get_num_threads(), " threads; julia threads: ", Threads.nthreads())
    commit = try
        strip(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
    println("* MixedModels at commit ", commit)
    println("* `RFPthreshold` = ", opts["rfpthreshold"], ", reps = ", opts["reps"])
    println("* ", length(models), " models: ", join((bm.name for bm in models), ", "))
    println()
    return nothing
end

function suitetable(shapes, models)
    println("## The model suite\n")
    println("Block types are those of `L`, lower triangle by rows, random-effects terms")
    println("only; the fixed-effects row is dense in every model.  `A/L` denotes a block")
    println("whose type differs between `A` and `L`.\n")
    rows = map(models) do bm
        d = shapes[bm.name]
        return [bm.name, string(bm.tier), d["dataset"], d["n"], d["p"], d["ntheta"],
            d["nre"], d["qs"], "`" * d["shape"] * "`", mib(int(d, "Lbytes")), bm.note]
    end
    mdtable(
        ["model", "tier", "dataset", "n", "p", "nθ", "#RE", "RE block sizes",
            "block types of `L`", "`L` (MiB)", "shape"],
        [:l, :l, :l, :r, :r, :r, :r, :r, :l, :r, :l],
        rows)
    return nothing
end

function evaltable(evals, models)
    println("## The cost of one evaluation at the optimum\n")
    println("`objective` is `updateL!` plus the profiled objective, exactly what a")
    println("derivative-free optimizer evaluates; both gradient columns include that same")
    println("work.  `rel. diff` is the largest relative discrepancy between the analytic")
    println("and the ForwardDiff gradient, and `ws` is the size of the reusable workspace")
    println("each gradient source allocates once per optimization.  On the smallest models")
    println("all three kernels are dominated by fixed per-call overhead rather than by")
    println("arithmetic, which is why ForwardDiff can come out ahead there.\n")
    rows = Vector{String}[]
    for bm in models
        haskey(evals, bm.name) || continue
        d = evals[bm.name]
        o, a, f = num(d, "objtime"), num(d, "gradtime"), num(d, "fdtime")
        push!(rows, [bm.name, d["ntheta"],
            ms(o), ms(a), ratio(a, o), ms(f), ratio(f, a),
            kib(int(d, "objalloc")), kib(int(d, "gradalloc")), kib(int(d, "fdalloc")),
            mib(int(d, "wsbytes")), mib(int(d, "fdwsbytes")),
            f < 0 ? "—" : @sprintf("%.1e", num(d, "reldiff"))])
    end
    mdtable(
        ["model", "nθ", "objective (ms)", "analytic ∇ (ms)", "∇/obj",
            "ForwardDiff ∇ (ms)", "FD/analytic", "obj alloc (KiB)", "∇ alloc (KiB)",
            "FD ∇ alloc (KiB)", "analytic ws (MiB)", "FD ws (MiB)", "rel. diff"],
        [:l, :r, :r, :r, :r, :r, :r, :r, :r, :r, :r, :r, :r],
        rows)
    return nothing
end

label(d) = startswith(d["optimizer"], "LD") ? d["optimizer"] * " + " * d["gradient"] :
           d["optimizer"]

function fittable(fits, models)
    println("## Complete fits\n")
    println("`Δobjective` is measured against the smallest objective reached for that")
    println("model, so a positive value means that configuration stopped short of the")
    println("best optimum found.  `max|∇|` is the analytic gradient at the returned")
    println("optimum on the deviance scale, a scale-free measure of how tightly each")
    println("optimizer converged.  ms/eval includes the gradient for the `LD_*` rows.\n")
    rows = Vector{String}[]
    for bm in models
        ds = filter(d -> d["model"] == bm.name, fits)
        isempty(ds) && continue
        bestfmin = minimum(d -> num(d, "fmin"), ds)
        for d in ds
            t, fe = num(d, "time"), int(d, "feval")
            push!(rows, [bm.name, d["ntheta"], label(d), d["effopt"], string(fe),
                sec(t), @sprintf("%.3f", 1000 * t / fe),
                mib(int(d, "bytes")), d["allocs"], mib(int(d, "maxrss")),
                @sprintf("%.4f", num(d, "fmin")),
                @sprintf("%.2e", num(d, "fmin") - bestfmin),
                @sprintf("%.1e", num(d, "maxg")),
                d["singular"] == "true" ? "yes" : "", d["ret"]])
        end
    end
    mdtable(
        ["model", "nθ", "configuration", "algorithm", "feval", "time (s)", "ms/eval",
            "alloc (MiB)", "# allocs", "peak RSS (MiB)", "objective", "Δobjective",
            "max\\|∇\\|", "singular", "return"],
        [:l, :r, :l, :l, :r, :r, :r, :r, :r, :r, :r, :r, :r, :l, :l],
        rows)
    return nothing
end

function summarytable(fits, models, shapes)
    println("## Summary: time to a complete fit\n")
    println("Each cell is the wall-clock time in seconds and, in parentheses, the speed-up")
    println("relative to the fastest derivative-free configuration for that model.  A")
    println("value below 1× means the gradient did not pay for itself.\n")
    labels = unique(label.(fits))
    rows = Vector{String}[]
    for bm in models
        ds = filter(d -> d["model"] == bm.name, fits)
        isempty(ds) && continue
        free = filter(d -> !startswith(d["optimizer"], "LD"), ds)
        base = isempty(free) ? NaN : minimum(d -> num(d, "time"), free)
        cells = map(labels) do lab
            i = findfirst(d -> label(d) == lab, ds)
            isnothing(i) && return "—"
            t = num(ds[i], "time")
            return isnan(base) ? sec(t) : @sprintf("%s (%.1f×)", sec(t), base / t)
        end
        d = shapes[bm.name]
        push!(rows, [bm.name, d["ntheta"], d["n"], cells...])
    end
    mdtable(["model", "nθ", "n", labels...],
        [:l, :r, :r, (:r for _ in labels)...], rows)
    return nothing
end

function listmodels()
    mdtable(["model", "tier", "dataset", "ForwardDiff", "shape"],
        [:l, :l, :l, :l, :l],
        [[bm.name, string(bm.tier), string(bm.dsname), bm.fd ? "yes" : "no", bm.note]
         for bm in MODELS])
    return nothing
end

function driver(args)
    opts = parseoptions(args)
    if flag(opts, "list")
        listmodels()
        return nothing
    end
    models = selectmodels(opts)
    isempty(models) && throw(ArgumentError("no models selected"))
    rfp = opts["rfpthreshold"]

    shapes = Dict{String,Dict{String,String}}()
    blocks = Pair{String,String}[]
    for bm in models
        @info "block structure" model = bm.name
        out = runworker(["shape", bm.name, rfp])
        shapes[bm.name] = only(collectlines(out, "SHAPE"))
        push!(blocks, bm.name => extractblocks(out))
    end

    evals = Dict{String,Dict{String,String}}()
    if flag(opts, "evals")
        for bm in models
            @info "evaluation cost" model = bm.name
            try
                out = runworker(["eval", bm.name, rfp, opts["seconds"]])
                lines = collectlines(out, "EVAL")
                if isempty(lines)
                    @warn "no EVAL line" model = bm.name
                else
                    evals[bm.name] = only(lines)
                end
            catch err
                @warn "evaluation benchmark failed" model = bm.name exception = err
            end
        end
    end

    fits = Dict{String,String}[]
    if flag(opts, "fits")
        for (bm, o, g) in configs(models, opts)
            @info "fit" model = bm.name optimizer = o gradient = g
            try
                out = runworker(["fit", bm.name, o, g, rfp, opts["reps"], opts["maxtime"]])
                lines = collectlines(out, "FIT")
                isempty(lines) ? @warn("no FIT line", model = bm.name, optimizer = o) :
                append!(fits, lines)
            catch err
                @warn "configuration failed" model = bm.name optimizer = o gradient = g exception = err
            end
        end
    end

    println()
    header(opts, models)
    suitetable(shapes, models)
    flag(opts, "evals") && evaltable(evals, models)
    if flag(opts, "fits") && !isempty(fits)
        fittable(fits, models)
        summarytable(fits, models, shapes)
    end
    if flag(opts, "shapes")
        println("## The block structure of each model\n")
        for (name, bd) in blocks
            println("### ", name, "\n")
            println("```")
            println(bd)
            println("```\n")
        end
    end
    return nothing
end

#####
##### entry point
#####

# `using` and macro calls have to be resolved when this file is parsed, so the
# packages a worker needs are decided before any of the work happens
const MODE = (!isempty(ARGS) && first(ARGS) in ("shape", "eval", "fit")) ? first(ARGS) :
             "driver"
const NEEDS_FD = (MODE == "eval" && getbenchmodel(ARGS[2]).fd) ||
                 (MODE == "fit" && ARGS[4] == "forwarddiff")

if NEEDS_FD
    using ForwardDiff
end

if MODE == "shape"
    worker_shape(ARGS[2], parse(Int, ARGS[3]))
elseif MODE == "eval"
    worker_eval(ARGS[2], parse(Int, ARGS[3]), parse(Float64, ARGS[4]))
elseif MODE == "fit"
    worker_fit(ARGS[2], ARGS[3], ARGS[4], parse(Int, ARGS[5]), parse(Int, ARGS[6]),
        length(ARGS) > 6 ? parse(Float64, ARGS[7]) : -1.0)
else
    driver(ARGS)
end
