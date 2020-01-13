using BenchmarkTools, DataFrames, MixedModels, RData, Tables

const SUITE = BenchmarkGroup()

const dat = Dict(Symbol(k) => v for (k, v) in load(joinpath(
    dirname(pathof(MixedModels)),
    "..",
    "test",
    "dat.rda",
)));

categorical!(dat[:ml1m], [:G,:H]);  # forgot to convert these grouping factors

const mods = Dict{Symbol,Vector{Expr}}(
    :Alfalfa => [:(1 + A * B + (1 | G)), :(1 + A + B + (1 | G))],
    :Animal => [:(1 + (1 | G) + (1 | H))],
    :Arabidopsis => [],              # glmm and rename variables
    :Assay => [:(1+A+B*C+(1|G)+(1|H))],
    :AvgDailyGain => [:(1 + A * U + (1 | G)), :(1 + A + U + (1 | G))],
    :BIB => [:(1 + A * U + (1 | G)), :(1 + A + U + (1 | G))],
    :Bond => [:(1 + A + (1 | G))],
    :Chem97 => [:(1 + (1 | G) + (1 | H)), :(1 + U + (1 | G) + (1 | H))],
    :Contraception => [],            # glmm and rename variables
    :Cultivation => [:(1 + A * B + (1 | G)), :(1 + A + B + (1 | G)), :(1 + A + (1 | G))],
    :Demand => [:(1 + U + V + W + X + (1 | G) + (1 | H))],
    :Dyestuff => [:(1 + (1 | G))],
    :Dyestuff2 => [:(1 + (1 | G))],
    :Early => [:(1 + U + U & A + (1 + U | G))],
    :Exam => [:(1 + A * U + B + (1 | G)), :(1 + A + B + U + (1 | G))],
    :Gasoline => [:(1 + U + (1 | G))],
    :Gcsemv => [:(1 + A + (1 | G))], # variables must be renamed
    :Genetics => [:(1 + A + (1 | G) + (1 | H))],
    :HR => [:(1 + A * U + V + (1 + U | G))],
    :Hsb82 => [:(1 + A + B + C + U + (1 | G))],
    :IncBlk => [:(1 + A + U +  + W + Z + (1 | G))],
    :InstEval => [:(1 + A + (1 | G) + (1 | H) + (1 | I)), :(1 + A * I + (1 | G) + (1 | H))],
    :KKL => [],                      # variables must be renamed
    :KWDYZ => [],                    # variables must be renamed
    :Mississippi => [:(1 + A + (1 | G))],
    :Mmmec => [],                    # glmm (and offset) and variables renamed
    :Multilocation => [:(1 + A + (0 + A | G) + (1 | H))],
    :Oxboys => [:(1 + U + (1 + U | G))],
    :PBIB => [:(1 + A + (1 | G))],
    :Pastes => [:(1 + (1 | G) + (1 | H))],
    :Penicillin => [:(1 + (1 | G) + (1 | H))],
    :Pixel => [:(1 + U + V + (1 + U | G) + (1 | H))],  # variables must be renamed
    :Poems => [:(1 + U + V + W + (1 | G) + (1 | H) + (1 | I))],
    :Rail => [:(1 + (1 | G))],
    :SIMS => [:(1 + U + (1 + U | G))],
    :ScotsSec => [:(1 + A + U + V + (1 | G) + (1 | H))],
    :Semi2 => [:(1 + A + (1 | G) + (1 | H))],
    :Semiconductor => [:(1 + A * B + (1 | G))],
    :Socatt => [],                   # variables must be renamed - binomial glmm?
    :TeachingII => [:(1 + A + T + U + V + W + X + Z + (1 | G))],
    :VerbAgg => [:(1 + A + B + C + U + (1 | G) + (1 | H))], # Bernoulli glmm and rename variables
    :Weights => [:(1 + A * U + (1 + U | G))],
    :WWheat => [:(1 + U + (1 + U | G))],
    :bdf => [],                      # rename variables and look up model
    :bs10 => [:(1 + U + V + W + ((1 + U + V + W) | G) + ((1 + U + V + W) | H))],
    :cake => [:(1 + A * B + (1 | G))],
    :cbpp => [:(1 + A + (1 | G))],   # Binomial glmm, create and rename variables
    :d3 => [
        :(1 + U + (1 | G) + (1 | H) + (1 | I)),
        :(1 + U + (1 + U | G) + (1 + U | H) + (1 + U | I)),
    ],
    :dialectNL => [:(1 + A + T + U + V + W + X + (1 | G) + (1 | H) + (1 | I))],
    :egsingle => [:(1 + A + U + V + (1 | G) + (1 | H))],
    :epilepsy => [],                 # unknown origin
    :ergoStool => [:(1 + A + (1 | G))],
    :gb12 => [:(1 + S + T + U + V + W + X + Z + ((1 + S + U + W) | G) +
                ((1 + S + T + V) | H))],
    :grouseticks => [],              # rename variables, glmm needs formula
    :guImmun => [],                  # rename variables, glmm needs formula
    :guPrenat => [],                 # rename variables, glmm needs formula
    :kb07 => [
        :(1 + S + T + U + V + W + X + Z + ((1 + S + T + U + V + W + X + Z) | G) +
          ((1 + S + T + U + V + W + X + Z) | H)),
        :(1 + S + T + U + V + W + X + Z +
          zerocorr((1 + S + T + U + V + W + X + Z) | G) +
          zerocorr((1 + S + T + U + V + W + X + Z) | H)),
    ],
    :ml1m => [:(1 + (1 | G) + (1 | H))],
    :paulsim => [:(1 + S + T + U + (1 | H) + (1 | G))],  # names of H and G should be reversed
    :sleepstudy => [:(1 + U + (1 + U | G)), :(1 + U + zerocorr(1 + U | G))],
    :s3bbx => [],                    # probably drop this one
    :star => [],                     # not sure it is worthwhile working with these data
);

fitbobyqa(rhs::Expr, dsname::Symbol) =
    fit(MixedModel, @eval(@formula(Y ~ $rhs)), dat[dsname])
compactstr(ds, rhs) = replace(string(ds, ':', rhs), ' ' => "")

SUITE["simplescalar"] = BenchmarkGroup(["single", "simple", "scalar"])
for ds in [
    :Alfalfa,
    :AvgDailyGain,
    :BIB,
    :Bond,
    :cake,
    :Cultivation,
    :Dyestuff,
    :Dyestuff2,
    :ergoStool,
    :Exam,
    :Gasoline,
    :Hsb82,
    :IncBlk,
    :Mississippi,
    :PBIB,
    :Rail,
    :Semiconductor,
    :TeachingII,
]
    for rhs in mods[ds]
        SUITE["simplescalar"][compactstr(ds, rhs)] = @benchmarkable fitbobyqa(
            $(QuoteNode(rhs)),
            $(QuoteNode(ds)),
        )
    end
end

SUITE["singlevector"] = BenchmarkGroup(["single", "vector"])
for ds in [:Early, :HR, :Oxboys, :SIMS, :sleepstudy, :Weights, :WWheat]
    for rhs in mods[ds]
        SUITE["singlevector"][compactstr(ds, rhs)] = @benchmarkable fitbobyqa(
            $(QuoteNode(rhs)),
            $(QuoteNode(ds)),
        )
    end
end

SUITE["nested"] = BenchmarkGroup(["multiple", "nested", "scalar"])
for ds in [:Animal, :Chem97, :Genetics, :Pastes, :Semi2]
    for rhs in mods[ds]
        SUITE["nested"][compactstr(ds, rhs)] = @benchmarkable fitbobyqa(
            $(QuoteNode(rhs)),
            $(QuoteNode(ds)),
        )
    end
end

SUITE["crossed"] = BenchmarkGroup(["multiple", "crossed", "scalar"])

for ds in [
    :Assay,
    :Demand,
    :InstEval,
    :Penicillin,
    :ScotsSec,
    :dialectNL,
    :egsingle,
    :ml1m,
    :paulsim,
]
    for rhs in mods[ds]
        SUITE["crossed"][compactstr(ds, rhs)] = @benchmarkable fitbobyqa(
            $(QuoteNode(rhs)),
            $(QuoteNode(ds)),
        )
    end
end

SUITE["crossedvector"] = BenchmarkGroup(["multiple", "crossed", "vector"])
for ds in [:bs10, :d3, :gb12, :kb07]
    for rhs in mods[ds]
        SUITE["crossedvector"][compactstr(ds, rhs)] = @benchmarkable fitbobyqa(
            $(QuoteNode(rhs)),
            $(QuoteNode(ds)),
        )
    end
end
