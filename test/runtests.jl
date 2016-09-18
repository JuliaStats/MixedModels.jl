using DataFrames, Feather, MixedModels, Base.Test

ds = Feather.read(joinpath(dirname(@__FILE__), "data", "Dyestuff.feather"))
ds2 = Feather.read(joinpath(dirname(@__FILE__), "data", "Dyestuff2.feather"))
slp = Feather.read(joinpath(dirname(@__FILE__), "data", "sleepstudy.feather"))
psts = Feather.read(joinpath(dirname(@__FILE__), "data", "Pastes.feather"))
pen = Feather.read(joinpath(dirname(@__FILE__), "data", "Penicillin.feather"))
cbpp = Feather.read(joinpath(dirname(@__FILE__), "data", "CBPP.feather"))
contra = Feather.read(joinpath(dirname(@__FILE__), "data", "Contraception.feather"))

include("paramlowertriangular.jl")
include("scalarReTerm.jl")
include("vectorReTerm.jl")
include("pls.jl")
include("pirls.jl")
include("throws.jl")
include("inject.jl")
include("cfactor.jl")
