using DataFrames, CSV, MixedModels, CairoMakie

df = CSV.read("issues/833/data.csv", DataFrame)

_lmm = LinearMixedModel(
    @formula(endpoint ~ 1 + formulation + sequence + period + (1 | id)),
    df;
    contrasts = Dict(:period => DummyCoding()),
)

_lmm.optsum.optimizer = :LN_COBYLA

fit!(_lmm; REML = true, fitlog=true)

θ = _lmm.θ    # keep a copy of the optimal θ
lines(0.4:0.01:1.0, x -> objective(updateL!(setθ!(_lmm, (x,)))))
setθ!(_lmm, θ)
