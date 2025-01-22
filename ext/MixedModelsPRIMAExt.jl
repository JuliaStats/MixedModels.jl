module MixedModelsPRIMAExt

using MixedModels: MixedModels, LinearMixedModel, objective!
using MixedModels: ProgressMeter, ProgressUnknown
using PRIMA: PRIMA

function MixedModels.prfit!(m::LinearMixedModel;
                            progress::Bool=true,
                            REML::Bool=m.optsum.REML,
                            σ::Union{Real,Nothing}=m.optsum.sigma,
                            thin::Int=1)
    optsum = m.optsum
    copyto!(optsum.final, optsum.initial)
    optsum.REML = REML
    optsum.sigma = σ

    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    # start from zero for the initial call to obj before optimization
    iter = 0
    fitlog = empty!(optsum.fitlog)
    function obj(x)
        iter += 1
        val = if isone(iter) && x == optsum.initial
            optsum.finitial
        else
            try
                objective!(m, x)
            catch ex
                # This can happen when the optimizer drifts into an area where
                # there isn't enough shrinkage. Why finitial? Generally, it will
                # be the (near) worst case scenario value, so the optimizer won't
                # view it as an optimum. Using Inf messes up the quadratic
                # approximation in BOBYQA.
                ex isa PosDefException || rethrow()
                optsum.finitial
            end
        end
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        !isone(iter) && iszero(rem(iter, thin)) && push!(fitlog, (copy(x), val))
        return val
    end

    info = PRIMA.bobyqa!(obj, optsum.final; xl=m.optsum.lowerbd)
    optsum.feval = info.nf
    optsum.fmin = info.fx
    optsum.returnvalue = Symbol(info.status)
    return m
end

end # module
