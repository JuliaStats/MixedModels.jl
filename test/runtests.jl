using DataArrays, DataFrames, Feather, MixedModels, Base.Test

const datadir = joinpath(dirname(@__FILE__), "data")
#const datadir = Pkg.dir("MixedModels", "test", "data")
cbpp = Feather.read(joinpath(datadir, "CBPP.feather"), nullable = false)
contraception = Feather.read(joinpath(datadir, "Contraception.feather"), nullable = false)
dyestuff = Feather.read(joinpath(datadir, "Dyestuff.feather"), nullable = false)
dyestuff2 = Feather.read(joinpath(datadir, "Dyestuff2.feather"), nullable = false)
insteval = Feather.read(joinpath(datadir, "InstEval.feather"), nullable = false)
pastes = Feather.read(joinpath(datadir, "Pastes.feather"), nullable = false)
penicillin = Feather.read(joinpath(datadir, "Penicillin.feather"), nullable = false)
sleepstudy = Feather.read(joinpath(datadir, "sleepstudy.feather"), nullable = false)

include("paramlowertriangular.jl")
include("scalarReTerm.jl")
#include("vectorReTerm.jl")
include("pls.jl")
include("pirls.jl")
#include("throws.jl")
