using MixedModels, RData

const dat = convert(Dict{Symbol,Any}, load(Pkg.dir("MixedModels", "test", "dat.rda")));

const mods1 = Dict{Symbol,Vector{String}}(  # models that can be benchmarked in 1 second
    :Alfalfa => ["1+A*B+(1|G)"],
    :Animal => ["1+(1|G)+(1|H)"],
    :Assay => ["1+A+B*C+(1|G)+(1|H)"],
    :AvgDailyGain => ["1+A*U+(1|G)", "1+A+U+(1|G)"],
    :BIB => ["1+A*U+(1|G)", "1+A+U+(1|G)"],
    :Bond => ["1+A+(1|G)"],
    :cake => ["1+A*B+(1|G)"],
    :Cultivation => ["1+A*B+(1|G)"],
    :Demand => ["1+U+V+W+X+(1|G)+(1|H)"],
    :Dyestuff => ["1+(1|G)"],
    :Dyestuff2 => ["1+(1|G)"],
    :ergoStool => ["1+A+(1|G)"],
    :Exam => ["1+A*U+B+(1|G)", "1+A+B+U+(1|G)"],
    :Gasoline => ["1+U+(1|G)"],
    :Genetics => ["1+A+(1|G)+(1|H)"],
    :HR => ["1+A*U+V+(1+U|G)"],
    :Hsb82 => ["1+A+B+C+U+(1|G)"],
    :IncBlk => ["1+A+U+V+W+Z+(1|G)"],
    :Mississippi => ["1+A+(1|G)"],
    :Multilocation => ["1+A+(0+A|G)+(1|H)"],
    :Oxboys => ["1+U+(1+U|G)"],
    :Pastes => ["1+(1|G)+(1|H)"],
    :PBIB => ["1+A+(1|G)"],
    :Penicillin => ["1+(1|G)+(1|H)"],
    :ScotsSec => ["1+A+U+V+(1|G)+(1|H)"],
    :Semi2 => ["1+A+(1|G)+(1|H)"],
    :Semiconductor => ["1+A*B+(1|G)"],
    :SIMS => ["1+U+(1+U|G)"],
    :sleepstudy => ["1+U+(1+U|G)", "1+U+(1|G)+(0+U|G)"],
    :TeachingII => ["1+A+T+U+V+W+X+Z+(1|G)"],
    :Weights => ["1+A*U+(1+U|G)"],
    :WWheat => ["1+U+(1+U|G)"]
    );

fitbobyqa(dsname::Symbol) =
    fit!(lmm(DataFrames.Formula(:Y, parse(mods1[dsname][1])), dat[dsname]))

@benchgroup "simplescalar" ["single", "simple", "scalar"] begin
    for ds in [:Alfalfa, :AvgDailyGain, :BIB, :Bond, :Cultivation, :Dyestuff,
        :Dyestuff2, :ergoStool, :Exam, :Gasoline, :Hsb82, :IncBlk, :Mississippi,
        :PBIB, :Semiconductor, :TeachingII]
        @bench string(ds) fitbobyqa($(QuoteNode(ds)))
    end
end

@benchgroup "singlevector" ["single", "vector"] begin
    for ds in [:HR, :Oxboys, :SIMS, :sleepstudy, :Weights, :WWheat]
        @bench string(ds) fitbobyqa($(QuoteNode(ds)))
    end
end

@benchgroup "nested" ["multiple", "nested", "scalar"] begin
    for ds in [:Animal, :Genetics, :Pastes, :Semi2]
        @bench string(ds) fitbobyqa($(QuoteNode(ds)))
    end
end

@benchgroup "crossed" ["multiple", "crossed", "scalar"] begin
    for ds in [:Assay, :Demand, :Multilocation, :Penicillin, :ScotsSec]
        @bench string(ds) fitbobyqa($(QuoteNode(ds)))
    end
end
