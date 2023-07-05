include("cache.jl")
using .Cache
using Arrow
using CategoricalArrays
using DataFrames
using Distributed
using MixedModels
using ProgressMeter
using Random
using StandardizedPredictors

kb07 = MixedModels.dataset(:kb07)
contrasts = Dict(:item => Grouping(),
    :subj => Grouping(),
    :spkr => EffectsCoding(),
    :prec => EffectsCoding(),
    :load => EffectsCoding())
m07 = fit(MixedModel,
    @formula(
        1000 / rt_raw ~
            1 + spkr * prec * load +
            (1 + spkr * prec * load | item) +
            (1 + spkr * prec * load | subj)
    ),
    kb07; contrasts, progress=true, thin=1)

pbref = @time parametricbootstrap(MersenneTwister(42), 1000, m07);
confint(pbref)
pb_restricted = @time parametricbootstrap(
    MersenneTwister(42), 1000, m07; optsum_overrides=(; ftol_rel=1e-3)
);
pb_restricted2 = @time parametricbootstrap(
    MersenneTwister(42), 1000, m07; optsum_overrides=(; ftol_rel=1e-6)
);
confint(pb_restricted)

optsum_overrides = (; maxfeval=round(Int, 0.9 * m07.optsum.feval))
@info "Currently using $(nprocs()) processors total and $(nworkers()) for work"

# you already have 1 proc by default, so add the number of additional cores
# you want to use, but see below
addprocs(3)
@info "Currently using $(nprocs()) processors total and $(nworkers()) for work"
rmprocs(workers())
@info "Currently using $(nprocs()) processors total and $(nworkers()) for work"

addprocs(3)
# load the necessary packages in other processes
@everywhere using MixedModels, Random
@info "Currently using $(nprocs()) processors total and $(nworkers()) for work"

# copy everything to workers
@showprogress for w in workers()
    remotecall_fetch(() -> coefnames(m07), w)
end

# you need at least as many RNGs as cores you want to use in parallel
# but you shouldn't use all of your cores because nested within this
# is the multithreading of the linear algebra
# 5 RNGS and 10 replicates from each
pb_map = @time @showprogress pmap(MersenneTwister.(41:45)) do rng
    parametricbootstrap(rng, 10, m07; optsum_overrides)
end;
@time confint(reduce(vcat, pb_map))
##################################################

using .Cache
using Distributed
addprocs(3)
@everywhere using MixedModels, Random, StandardizedPredictors
df = DataFrame(Arrow.Table(Cache.data_path()))

transform!(df, :Sex => categorical, :Test => categorical; renamecols=false)
recode!(df.Test,
    "Run" => "Endurance",
    "Star_r" => "Coordination",
    "S20_r" => "Speed",
    "SLJ" => "PowerLOW",
    "BPT" => "PowerUP")
df = combine(groupby(df, :Test), :, :score => zscore => :zScore)
describe(df)

contrasts = Dict(:Cohort => Grouping(),
    :School => Grouping(),
    :Child => Grouping(),
    :Test => SeqDiffCoding(),
    :Sex => EffectsCoding(),
    :age => Center(8.5))

f1 = @formula(
    zScore ~
        1 + age * Test * Sex +
        (1 + Test + age + Sex | School) +
        (1 + Test | Child) +
        zerocorr(1 + Test | Cohort)
)
m1 = fit(MixedModel, f1, df; contrasts, progress=true, thin=1)

# copy everything to workers
@showprogress for w in workers()
    remotecall_fetch(() -> coefnames(m1), w)
end

# you need at least as many RNGs as cores you want to use in parallel
# but you shouldn't use all of your cores because nested within this
# is the multithreading of the linear algebra
# 5 RNGS and 10 replicates from each
pb_map = @time @showprogress pmap(MersenneTwister.(41:45)) do rng
    parametricbootstrap(rng, 100, m1; optsum_overrides=(; maxfeval=300))
end;
@time confint(reduce(vcat, pb_map))

# dev = last.(m1.optsum.fitlog)

# julia> findlast(x -> x / last(dev) > 1.1, dev)
# 50

# julia> findlast(x -> x / last(dev) > 1.01, dev)
# 100

# julia> findlast(x -> x / last(dev) > 1.001, dev)
# 296

# julia> findlast(x -> x / last(dev) > 1.0001, dev)
# 1805

# julia> findlast(x -> x / last(dev) > 1.00001, dev)
# 3902

# julia> findlast(x -> x / last(dev) > 1.000001, dev)
# 4783

# julia> findlast(x -> x / last(dev) > 1.0000001, dev)
# 5352

# julia> findlast(x -> x / last(dev) > 1.00000001, dev)
# 5539

# julia> findlast(x -> x / last(dev) > 1.000000001, dev)
# 5565

# julia> findlast(x -> x / last(dev) > 1.0000000001, dev)
# 5566

# julia> findlast(x -> x / last(dev) > 1.00000000001, dev)
# 5566
