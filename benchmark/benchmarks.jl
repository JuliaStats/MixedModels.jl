using BenchmarkTools, MixedModels
using MixedModels: dataset

const SUITE = BenchmarkGroup()

const global contrasts = Dict{Symbol,Any}(
    :batch => Grouping(),     # dyestuff, dyestuff2, pastes
    :cask => Grouping(),      # pastes
    :d => Grouping(),         # insteval
    # :dept => Grouping(),    # insteval - not set b/c also used in fixed effects
    :g => Grouping(),         # d3, ml1m
    :h => Grouping(),         # d3, ml1m
    :i => Grouping(),         # d3
    :item => Grouping(),      # kb07, mrk17_exp1,
    :Machine => Grouping(),   # machines
    :plate => Grouping(),     # penicillin
    :s => Grouping(),         # insteval
    :sample => Grouping(),    # penicillin
    :subj => Grouping(),      # kb07, mrk17_exp1, sleepstudy
    :Worker => Grouping(),    # machines
    :F => HelmertCoding(),       # mrk17_exp1
    :P => HelmertCoding(),       # mrk17_exp1
    :Q => HelmertCoding(),       # mrk17_exp1
    :lQ => HelmertCoding(),      # mrk17_exp1
    :lT => HelmertCoding(),      # mrk17_exp1
    :load => HelmertCoding(),    # kb07
    :prec => HelmertCoding(),    # kb07
    :service => HelmertCoding(), # insteval
    :spkr => HelmertCoding(),    # kb07
)

const global fms = Dict(
    :dyestuff => [
        @formula(yield ~ 1 + (1 | batch))
    ],
    :dyestuff2 => [
        @formula(yield ~ 1 + (1 | batch))
    ],
    :d3 => [
        @formula(y ~ 1 + u + (1 + u | g) + (1 + u | h) + (1 + u | i))
    ],
    :insteval => [
        @formula(y ~ 1 + service + (1 | s) + (1 | d) + (1 | dept)),
        @formula(y ~ 1 + service * dept + (1 | s) + (1 | d)),
    ],
    :kb07 => [
        @formula(rt_trunc ~ 1 + spkr + prec + load + (1 | subj) + (1 | item)),
        @formula(rt_trunc ~ 1 + spkr * prec * load + (1 | subj) + (1 + prec | item)),
        @formula(
            rt_trunc ~
                1 + spkr * prec * load + (1 + spkr + prec + load | subj) +
                (1 + spkr + prec + load | item)
        ),
    ],
    :machines => [
        @formula(score ~ 1 + (1 | Worker) + (1 | Machine))
    ],
    :ml1m => [
        @formula(y ~ 1 + (1 | g) + (1 | h))
    ],
    :mrk17_exp1 => [
        @formula(1000 / rt ~ 1 + F * P * Q * lQ * lT + (1 | item) + (1 | subj)),
        @formula(
            1000 / rt ~
                1 + F * P * Q * lQ * lT + (1 + P + Q + lQ + lT | item) +
                (1 + F + P + Q + lQ + lT | subj)
        ),
    ],
    :pastes => [
        @formula(strength ~ 1 + (1 | batch & cask)),
        @formula(strength ~ 1 + (1 | batch / cask)),
    ],
    :penicillin => [
        @formula(diameter ~ 1 + (1 | plate) + (1 | sample))
    ],
    :sleepstudy => [
        @formula(reaction ~ 1 + days + (1 | subj)),
        @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
        @formula(reaction ~ 1 + days + (1 | subj) + (0 + days | subj)),
        @formula(reaction ~ 1 + days + (1 + days | subj)),
    ],
)

function fitbobyqa(dsnm::Symbol, i::Integer)
    return fit(MixedModel, fms[dsnm][i], dataset(dsnm); contrasts, progress=false)
end

# these tests are so fast that they can be very noisy because the denominator is so small,
# so we disable them by default for auto-benchmarking
# SUITE["simplescalar"] = BenchmarkGroup(["single", "simple", "scalar"])
# for (ds, i) in [
#     (:dyestuff, 1),
#     (:dyestuff2, 1),
#     (:pastes, 1),
#     (:sleepstudy, 1),
# ]
#     SUITE["simplescalar"][string(ds, ':', i)] = @benchmarkable fitbobyqa($ds, $i)
# end

SUITE["singlevector"] = BenchmarkGroup(["single", "vector"])
for (ds, i) in [
    (:sleepstudy, 2),
    (:sleepstudy, 3),
    (:sleepstudy, 4),
]
    SUITE["singlevector"][string(ds, ':', i)] = @benchmarkable fitbobyqa($ds, $i)
end

SUITE["nested"] = BenchmarkGroup(["multiple", "nested", "scalar"])
for (ds, i) in [
(:pastes, 2)
]
    SUITE["nested"][string(ds, ':', i)] = @benchmarkable fitbobyqa($ds, $i)
end

SUITE["crossed"] = BenchmarkGroup(["multiple", "crossed", "scalar"])
for (ds, i) in [
    (:insteval, 1),
    (:insteval, 2),
    (:kb07, 1),
    (:machines, 1),
    (:ml1m, 1),
    (:mrk17_exp1, 1),
    (:penicillin, 1),
]
    SUITE["crossed"][string(ds, ':', i)] = @benchmarkable fitbobyqa($ds, $i)
end

SUITE["crossedvector"] = BenchmarkGroup(["multiple", "crossed", "vector"])
for (ds, i) in [
    (:d3, 1),
    (:kb07, 2),
    (:kb07, 3),
    (:mrk17_exp1, 2),
]
    SUITE["crossedvector"][string(ds, ':', i)] = @benchmarkable fitbobyqa($ds, $i)
end
