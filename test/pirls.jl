using Base.Test, DataFrames, MixedModels

contra = readtable(joinpath(dirname(@__FILE__), "data", "Contraception.csv.gz"))
for c in [:district, :use, :urban, :livch, :urbdist]
    contra[c] = pool(contra[c])
end
contra[:age2] = abs2(contra[:age])
gm1 = fit!(glmm(use01 ~ 1 + age + age2 + urban + livch + (1 | urbdist), contra, Binomial()));

@test isapprox(LaplaceDeviance(gm1), 2361.5457541; atol = 0.0001)
# There may be multiple optima here.
#@test isapprox(logdet(gm1), 75.7204822; atol = 0.0001)
#@test isapprox(sumabs2(gm1.u[1]), 48.47486965; atol = 0.0001)
#@test isapprox(sum(gm1.devresid), 2237.3504024; atol = 0.0001)

cbpp = readtable(joinpath(dirname(@__FILE__), "data", "cbpp.csv.gz"))
for c in [:herd, :period]
    cbpp[c] = pool(oftype(Int8[], cbpp[c]))
end
for c in [:incidence, :size]
  cbpp[c] = oftype(Int8[], cbpp[c])
end
cbpp[:prop] = cbpp[:incidence] ./ cbpp[:size]
gm2 = fit!(glmm(prop ~ 1 + period + (1 | herd), cbpp, Binomial(), LogitLink(); wt = cbpp[:size]));

@test isapprox(LaplaceDeviance(gm2), 100.095856; atol = 0.0001)
@test isapprox(sumabs2(gm2.u[1]), 9.72305; atol = 0.0001)
@test isapprox(logdet(gm2), 16.901054; atol = 0.0001)
@test isapprox(sum(gm2.devresid), 73.47143; atol = 0.001)
#VerbAgg = rcopy("lme4::VerbAgg")
#VerbAgg[:r201] = [Int8(x == "N" ? 0 : 1) for x in VerbAgg[:r2]]
#m3 = glmm(r201 ~ 1 + Anger + Gender + btype + situ + (1 | id) + (1 | item), VerbAgg, Bernoulli());
#f = r201 ~ 1 + Anger + Gender + btype + situ + (1 | id)
