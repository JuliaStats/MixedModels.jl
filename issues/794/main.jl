using CSV
using MixedModels
using MixedModelsDatasets: dataset
x = CSV.read(download("https://gist.githubusercontent.com/yjunechoe/af4a8dd3e2b0343fe3818e93dba8c81d/raw/x.csv"), Table);
m = fit(MixedModel, @formula(y ~ x1 * x2 + (1 | g)), x)

profile(m; threshold=4) # works

profile(m; threshold=3)

s = lmm(@formula(reaction ~ 1 + days + (1 + days| subj)), dataset(:sleepstudy))

profile(s; threshold=3)
