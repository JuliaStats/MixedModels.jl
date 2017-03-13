using DataArrays, DataFrames, MixedModels, RData, Base.Test

const dat = load(joinpath(dirname(@__FILE__), "dat.rda"))

const cbpp = dat["cbpp"]
const contraception = dat["Contraception"]
const dyestuff = dat["Dyestuff"]
const dyestuff2 = dat["Dyestuff2"]
const insteval = dat["InstEval"]
const pastes = dat["Pastes"]
const penicillin = dat["Penicillin"]
const sleepstudy = dat["sleepstudy"]

include("paramlowertriangular.jl")
include("scalarReTerm.jl")
include("vectorReTerm.jl")
include("pls.jl")
include("pirls.jl")
#include("throws.jl")
