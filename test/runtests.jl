using DataFrames,MixedModels
using Base.Test

include("data.jl")

tests = ["plsdiag.jl",
         #"plsgeneral.jl",
         "plsonescalar.jl",
         "plsonevector.jl",
         "plstwoscalar.jl",
         "plstwovector.jl",
         "plsdiagnested.jl"]

anyerrors = false

for t in tests
    try
        include(t)
        println("\t\033[1m\033[32mPASSED\033[0m: $t")
    catch
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $t")
    end
end

if anyerrors
    throw("Tests failed")
end
