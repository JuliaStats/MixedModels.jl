# This test file is not run with the package tests.
# It simply defines a function that can be used to compare the PRIMA and NLopt implementations of BOBYQA.
using MixedModels, PRIMA

include("modelcache.jl")
@isdefined(contrasts) || const contrasts = Dict{Symbol,Any}()
contrasts[:spkr] = HelmertCoding()
contrasts[:load] = HelmertCoding()
contrasts[:prec] = HelmertCoding()
contrasts[:service] = HelmertCoding()


function comparePRIMA(dsnm)
    res = @NamedTuple{ds::Symbol, Pfinal::Float64, Nfinal::Float64, Peval::Int64, Neval::Int64}[]
    if haskey(fms, dsnm)
        for f in fms[dsnm]
            m = LinearMixedModel(f, dataset(dsnm); contrasts)
            x, info = bobyqa(objective!(m), m.optsum.initial; xl=m.optsum.lowerbd)
            fit!(m; progress=false)
            push!(
                res,
                (;
                    ds = dsnm,
                    Pfinal=info.fx,
                    Nfinal=m.optsum.fmin,
                    Peval=info.nf,
                    Neval=m.optsum.feval,
                )
            )
        end
    end
    return res
end

