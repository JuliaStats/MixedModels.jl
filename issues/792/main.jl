using MixedModels, DataFrames, Random
Random.seed!(0)
df = allcombinations(DataFrame,
                     "subject" => 1:100,
                     "session" => 1:6,
                     "list" => 1:25,
                     "serialpos" => 1:12)
df[!, :recalled] = rand(nrow(df))
df[!, :recalled] = df[!, :recalled] .> .5

serialpos_contrasts = Dict(:serialpos => DummyCoding())

form = @formula(recalled ~ serialpos + zerocorr(fulldummy(serialpos) | subject) + (1 | subject & session))
m = MixedModel(
        form,
        df, Bernoulli();
        contrasts=serialpos_contrasts)
