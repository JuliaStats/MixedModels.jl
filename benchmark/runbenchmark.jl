using BenchmarkTools, MixedModels, RData

const dat = convert(Dict{Symbol,Any}, load(Pkg.dir("MixedModels", "test", "dat.rda")));

const mods = Dict{Symbol,Vector{String}}(
    :Alfalfa => ["1+A*B+(1|G)"],
    :Animal => ["1+(1|G)+(1|H)"],
    :Assay => ["1+A+B*C+(1|G)+(1|H)"],
    :AvgDailyGain => ["1+A*U+(1|G)", "1+A+U+(1|G)"],
    :BIB => ["1+A*U+(1|G)", "1+A+U+(1|G)"],
    :Bond => ["1+A+(1|G)"],
    :bs10 => ["1+U+V+W+(1+U+V+W|G)+(1+U+V+W|H)",
        "1+U+V+W++(1|G)+((0+U)|G)+((0+V)|G)+((0+W)|G)+(1|H)+((0+U)|H)+((0+V)|H)+((0+W)|H)"],
    :cake => ["1+A*B+(1|G)"]
    );

function fitlmm(dsname, form, opt)
    m = lmm(DataFrames.Formula(:Y, parse(form)), dat[Symbol(dsname)])
    m.optsum.optimizer = Symbol(opt)
    fit!(m)
end

const suite = BenchmarkGroup()
for (k,v) in mods
    suite[k] = BenchmarkGroup()
    for vi in v, opt in [:LN_BOBYQA, :LN_COBYLA, :LN_SBPLX]
        suite[k][vi,opt] = @benchmarkable fitlmm($k, $vi, $opt)
    end
end
tune!(suite)
