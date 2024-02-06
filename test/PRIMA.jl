# This test file is not run with the package tests.
# Compare the PRIMA and NLopt implementations of BOBYQA.

using DataFrames, MixedModels, PRIMA

include("modelcache.jl")
@isdefined(contrasts) || const contrasts = Dict{Symbol,Any}()
contrasts[:spkr] = HelmertCoding()
contrasts[:load] = HelmertCoding()
contrasts[:prec] = HelmertCoding()
contrasts[:service] = HelmertCoding()


function comparePRIMA(dsnm)
    res = @NamedTuple{ds::Symbol, frm::Int, k::Int, knn::Int, Pfinal::Float64, fdiff::Float64, Peval::Int, ediff::Int}[]
    if haskey(fms, dsnm)
        for (i, f) in enumerate(fms[dsnm])
            m = LinearMixedModel(f, dataset(dsnm); contrasts)
            xl = m.optsum.lowerbd
            x, info = bobyqa(objective!(m), m.optsum.initial; xl)
            fit!(m; progress=false)
            push!(
                res,
                (;
                    ds=dsnm,
                    frm=i,
                    k = length(xl),
                    knn = count(iszero, xl),
                    Pfinal=info.fx,
                    fdiff=m.optsum.fmin - info.fx,
                    Peval=info.nf,
                    ediff=m.optsum.feval - info.nf,
                )
            )
        end
    end
    return res
end

res = comparePRIMA(:dyestuff2)
for k in keys(fms)
    k == :dyestuff2 || append!(res, comparePRIMA(k))
end

println(DataFrame(sort(res, by=getproperty(:ediff))))
