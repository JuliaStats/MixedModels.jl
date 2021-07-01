using StatsModels
using MixedModels
slp = MixedModels.dataset(:sleepstudy)
f = @formula(reaction ~ 1 + (1|subj))
sch = schema(f, slp)
fsc = apply_schema(f, sch)
modelcols(fsc, slp)
