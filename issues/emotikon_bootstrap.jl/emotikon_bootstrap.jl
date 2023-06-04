module Cache
    using Downloads
    using Scratch
    # This will be filled in inside `__init__()`
    download_cache = ""
    url = "https://github.com/RePsychLing/SMLP2022/raw/main/data/fggk21.arrow"
    #"https://github.com/bee8a116-0383-4365-8df7-6c6c8d6c1322"

    function data_path()
        fname = joinpath(download_cache, basename(url))
        if !isfile(fname)
            @info "Local cache not found, downloading"
            Downloads.download(url, fname)
        end
        return fname
    end

    function __init__()
        global download_cache = get_scratch!(@__MODULE__, "downloaded_files")
        return nothing
    end
end

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
          @formula(1000 / rt_raw ~ 1 + spkr * prec * load +
                                  (1 + spkr * prec * load | item) +
                                  (1 + spkr * prec * load | subj)),
          kb07; contrasts, progress=true, thin=1)

pbref = @time parametricbootstrap(MersenneTwister(42), 50, m07);
confint(pbref)
pb_evalcap = @time parametricbootstrap(MersenneTwister(42), 50, m07; optsum_overrides=(;maxfeval=2000));
confint(pb_evalcap)

optsum_overrides = (;maxfeval=2000)
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
pb = @time @showprogress pmap(MersenneTwister.(40:45)) do rng
    parametricbootstrap(rng, 10, m07; optsum_overrides)
end;
@time confint(reduce(vcat, pb))



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

f1 = @formula(zScore ~ 1 + age * Test * Sex +
                      (1 + Test + age + Sex | School) +
                      (1 + Test | Child) +
              zerocorr(1 + Test | Cohort))
m1 = fit(MixedModel, f1, df; contrasts, progress=true, thin=1)
    # not sure this next call makes sense - should the second argument be m.optsum.final?
    _copy_away_from_lowerbd!(
        mnew.optsum.initial, mnew.optsum.final, mnew.lowerbd; incr=0.05
    )
