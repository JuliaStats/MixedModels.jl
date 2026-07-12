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
pb_restricted = @time parametricbootstrap(
    MersenneTwister(42), 1000, m07; optsum_overrides=(; ftol_rel=1e-3)
);
pb_restricted2 = @time parametricbootstrap(
    MersenneTwister(42), 1000, m07; optsum_overrides=(; ftol_rel=1e-6)
);
confint(pbref)
confint(pb_restricted)
confint(pb_restricted2)

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
