using MixedModels, CSV, DataFrames, StatsBase, Random
using SparseArrays: nzrange
using ProgressMeter

saddlepointdata = CSV.read("saddlepointdata.csv", DataFrame)
transform!(saddlepointdata, "AUCT" => eachindex => "row")

N = 500
_rng = Random.seed!(Random.default_rng(), 124)
formula = @formula(log(AUCT) ~ trt + seq + per +
            (trt + 0 | sub) +
                   zerocorr(trt + 0 | row))
contrasts = Dict(:trt => DummyCoding(), :per => DummyCoding(),
                 :sub => Grouping(), :row => Grouping())
fts = @showprogress map(1:N) do _
    data = transform(saddlepointdata,
                     "AUCT" => ByRow(t -> t + randn(_rng)*1e-12) => "AUCT")
    model = LinearMixedModel(formula, data; contrasts)
    model.optsum.optimizer = :LN_COBYLA
    return fit!(model; REML=true)
end

countmap(round.(exp.(getindex.(coef.(fts), 2)), digits=3))

using CairoMakie
saddlepointfit = fts[findfirst(t -> exp(coef(t)[2]) < 0.931, fts)];

plotx = range(-12, stop=12, length=101)
_i = 5
_θ = copy(saddlepointfit.θ)
ploty = [
    MixedModels.deviance(
        MixedModels.updateL!(
            MixedModels.setθ!(
                deepcopy(saddlepointfit), # to avoid mutating the original version
                _θ + [zeros(_i - 1); _x; zeros(5 - _i)]
            )
        )
    )
    for _x in plotx
];
lines(plotx, ploty)
