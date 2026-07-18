using PRIMA, MixedModels, CSV, DataFrames, StatsBase, Random

saddlepointdata = CSV.read("saddlepointdata.csv", DataFrame)
transform!(saddlepointdata, "AUCT" => eachindex => "row")

N = 500
_rng = Random.seed!(Random.default_rng(), 124)
fts = map(1:N) do _
    m = LinearMixedModel(
        @formula(log(AUCT) ~
            trt +
            seq +
            per +
            (trt + 0 | sub) +
            zerocorr(trt + 0 | row)
        ),
        transform(
            saddlepointdata,
            "AUCT" => ByRow(t -> t + randn(_rng)*1e-12) => "AUCT"
        );
        contrasts = Dict(
            [
                :trt,
                :per,
                :sub,
                :row,
            ] .=> Ref(DummyCoding())
        )
    )
    m.optsum.backend = :prima
    # m.optsum.optimizer = :cobyla
    m.optsum.optimizer = :bobyqa
    fit!(
        m;
        REML = true
    )
end

countmap(round.(exp.(getindex.(coef.(fts), 2)), digits=3))
