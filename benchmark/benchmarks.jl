using DataFrames, RData, MixedModels

const dat = convert(Dict{Symbol,DataFrame}, load(Pkg.dir("MixedModels", "test", "dat.rda")));

const mods = Dict{Symbol,Vector{Expr}}(
    :Alfalfa => [:(1+A*B+(1|G)), :(1+A+B+(1|G))],
    :Animal => [:(1+(1|G)+(1|H))],
    :Arabidopsis => [],             # glmm and rename variables
    :Assay => [:(1+A+B*C+(1|G)+(1|H))],
    :AvgDailyGain => [:(1+A*U+(1|G)), :(1+A+U+(1|G))],
    :BIB => [:(1+A*U+(1|G)), :(1+A+U+(1|G))],
    :Bond => [:(1+A+(1|G))],
    :Chem97 => [:(1q+(1|G)+(1|H)),:(1+U+(1|G)+(1|H))],
    :Contraception => [],           # glmm and rename variables
    :Cultivation => [:(1+A*B+(1|G)), :(1+A+B+(1|G)), :(1+A+(1|G))],
    :Demand => [:(1+U+V+W+X+(1|G)+(1|H))],
    :Dyestuff => [:(1+(1|G))],
    :Dyestuff2 => [:(1+(1|G))],
    :Early => [:(1+U+U&A+(1+U|G))], # variables must be renamed
    :Exam => [:(1+A*U+B+(1|G)), :(1+A+B+U+(1|G))],
    :Gasoline => [:(1+U+(1|G))],
    :Gcsemv => [:(1+A+(1|G))],      # variables must be renamed
    :Genetics => [:(1+A+(1|G)+(1|H))],
    :HR => [:(1+A*U+V+(1+U|G))],
    :Hsb82 => [:(1+A+B+C+U+(1|G))],
    :IncBlk => [:(1+A+U+V+W+Z+(1|G))],
    :InstEval => [:(1+A+(1|G)+(1|H)+(1|I)),:(1+A*I+(1|G)+(1|H))],
    :KKL => [],                    # variables must be renamed
    :KWDYZ => [],                  # variables must be renamed
    :Mississippi => [:(1+A+(1|G))],
    :Mmmec => [],                  # glmm (and offset) and variables renamed
    :Multilocation => [:(1+A+(0+A|G)+(1|H))],
    :Oxboys => [:(1+U+(1+U|G))],
    :PBIB => [:(1+A+(1|G))],
    :Pastes => [:(1+(1|G)+(1|H))],
    :Penicillin => [:(1+(1|G)+(1|H))],
    :Pixel => [:(1+U+V+(1+U|G)+(1|H))],  # variables must be renamed
    :Poems => [:(1+U+V+W+(1|G)+(1|H)+(1|I))],
    :Rail => [:(1+(1|G))],         # variables must be renamed
    :SIMS => [:(1+U+(1+U|G))],
    :ScotsSec => [:(1+A+U+V+(1|G)+(1|H))],
    :Semi2 => [:(1+A+(1|G)+(1|H))],
    :Semiconductor => [:(1+A*B+(1|G))],
    :Socatt => [],                 # variables must be renamed - binomial glmm?
    :TeachingII => [:(1+A+T+U+V+W+X+Z+(1|G))],
    :VerbAgg => [:(1+A+B+C+U+(1|G)+(1|H))], # Bernoulli glmm and rename variables
    :Weights => [:(1+A*U+(1+U|G))],
    :WWheat => [:(1+U+(1+U|G))],
    :bdf => [],                   # rename variables and look up model
    :bs10 => [:(1+U+V+W+((1+U+V+W)|G)+((1+U+V+W)|H))],
    :cake => [:(1+A*B+(1|G))],
    :cbpp => [:(1+A+(1|G))],      # Binomial glmm, create and rename variables
    :d3 => [:(1+U+(1|G)+(1|H)+(1|I)), :(1+U+(1+U|G)+(1+U|H)+(1+U|I))],
    :dialectNL => [:(1+A+T+U+V+W+X+(1|G)+(1|H)+(1|I))],
    :egsingle => [:(1+A+U+V+(1|G)+(1|H))],
    :epilepsy => [],              # unknown origin
    :ergoStool => [:(1+A+(1|G))],
    :gb12 => [:(1+S+T+U+V+W+X+Z+((1+S+U+W)|G)+((1+S+T+V)|H))],
    :grouseticks => [],           # rename variables, glmm needs formula
    :guimmun => [],               # rename variables, glmm needs formula
    :guPrenat => [],              # rename variables, glmm needs formula
    :kb07 => [:(1+S+T+U+V+W+X+Z+((1+S+T+U+V+W+X+Z)|G)+((1+S+T+U+V+W+X+Z)|H)),
              :(1+S+T+U+V+W+X+Z+(1|G)+((0+S)|G)+((0+T)|G)+((0+U)|G)+((0+V)|G)+((0+W)|G)+
              ((0+X)|G)+((0+Z)|G)+(1|H)+((0+S)|H)+((0+T)|H)+((0+U)|H)+((0+V)|H)+
              ((0+W)|H)+((0+X)|H)+((0+Z)|H))],
    :paulsim => [:(1+S+T+U+(1|H)+(1|G))],  # names of H and G should be reversed
    :sleepstudy => [:(1+U+(1+U|G)), :(1+U+(1|G)+(0+U|G))],
    :s3bbx => [],                 # probably drop this one
    :star => []                   # not sure it is worthwhile working with these data
    );

fitbobyqa(rhs::Expr, dsname::Symbol) = fit!(lmm(DataFrames.Formula(:Y, rhs), dat[dsname]))

@benchgroup "simplescalar" ["single", "simple", "scalar"] begin
    for ds in [:Alfalfa, :AvgDailyGain, :BIB, :Bond, :cake, :Cultivation, :Dyestuff,
        :Dyestuff2, :ergoStool, :Exam, :Gasoline, :Hsb82, :IncBlk, :Mississippi,
        :PBIB, :Semiconductor, :TeachingII]
        for rhs in mods[ds]
            @bench string(ds, ':', rhs) fitbobyqa($(QuoteNode(rhs)), $(QuoteNode(ds)))
        end
    end
end

@benchgroup "singlevector" ["single", "vector"] begin
    for ds in [:HR, :Oxboys, :SIMS, :sleepstudy, :Weights, :WWheat]
        for rhs in mods[ds]
            @bench string(ds, ':', rhs) fitbobyqa($(QuoteNode(rhs)), $(QuoteNode(ds)))
        end
    end
end

@benchgroup "nested" ["multiple", "nested", "scalar"] begin
    for ds in [:Animal, :Genetics, :Pastes, :Semi2]
        for rhs in mods[ds]
            @bench string(ds, ':', rhs) fitbobyqa($(QuoteNode(rhs)), $(QuoteNode(ds)))
        end
    end
end

@benchgroup "crossed" ["multiple", "crossed", "scalar"] begin
    for ds in [:Assay, :Demand, :InstEval, :Penicillin, :ScotsSec, :d3,
               :dialectNL, :egsingle, :paulsim]
        for rhs in mods[ds]
            @bench string(ds, ':', rhs) fitbobyqa($(QuoteNode(rhs)), $(QuoteNode(ds)))
        end
    end
end

@benchgroup "crossedvector" ["multiple", "crossed", "vector"] begin
    for ds in [:bs10, :gb12]
        for rhs in mods[ds]
            @bench string(ds, ':', rhs) fitbobyqa($(QuoteNode(rhs)), $(QuoteNode(ds)))
        end
    end
end
