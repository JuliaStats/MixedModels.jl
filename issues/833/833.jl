# Not sure I understand the purpose of this example as it doesn't evaluate the objective at negative θ values
using DataFrames, CSV, MixedModels, CairoMakie

df = CSV.read("issues/833/data.csv", DataFrame)

_lmm = LinearMixedModel(
    @formula(endpoint ~ 1 + formulation + sequence + period + (1 | id)),
    df;
    contrasts = Dict(:period => DummyCoding()),
)

#_lmm.optsum.optimizer = :LN_COBYLA  # use default of :LN_BOBYQA (1-dim optimization) instead

fit!(_lmm; REML = true)

lines(0.4:0.01:1.0, objective!(_lmm)) # I wrote the method for objective! then forgot it 
objective!(_lmm, _lmm.optsum.final)

_lmm.optsum.fitlog
