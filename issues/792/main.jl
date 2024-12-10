using MixedModels, DataFrames, StableRNGs
rng = StableRNG(0)
df = allcombinations(DataFrame,
                     "subject" => 1:10,
                     "session" => 1:6,
                     "serialpos" => 1:12)
df[!, :recalled] = rand(rng, [0, 1], nrow(df))
serialpos_contrasts = Dict(:serialpos => DummyCoding())

form = @formula(recalled ~ serialpos + zerocorr(serialpos | subject) + (1 | subject & session))
m = GeneralizedLinearMixedModel(form, df, Bernoulli(); contrasts=serialpos_contrasts);
m.optsum.ftol_rel = 1e-7
fit!(m; init_from_lmm=[:β, :θ], fast=true)
