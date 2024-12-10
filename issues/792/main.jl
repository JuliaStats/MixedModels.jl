using MixedModels, DataFrames, Random
Random.seed!(0)
df = allcombinations(DataFrame,
                     "subject" => 1:10,
                     "session" => 1:6,
                     "list" => 1:25,
                     "serialpos" => 1:12)
df[!, :recalled] = rand(nrow(df)) .> .5
serialpos_contrasts = Dict(:serialpos => DummyCoding())

form = @formula(recalled ~ serialpos + zerocorr(serialpos | subject) + (1 | subject & session))
m = GeneralizedLinearMixedModel(form, df, Bernoulli(); contrasts=serialpos_contrasts)
# m = GeneralizedLinearMixedModel(form, df, Bernoulli();
#                                 contrasts=serialpos_contrasts)

fit!(m; init_from_lmm=[:β, :θ])
