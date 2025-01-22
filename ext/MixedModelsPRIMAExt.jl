module MixedModelsPRIMAExt

using MixedModels: MixedModels, LinearMixedModel, objective!
using PRIMA: PRIMA

function MixedModels.prfit!(m::LinearMixedModel)
    θ, info = PRIMA.bobyqa(objective!(m), copy(m.optsum.initial); xl=m.optsum.lowerbd)
    if θ ≠ m.θ   # force an evaluation of the objective at the optimum
        objective!(m, θ)
    end
    m.optsum.feval = info.nf
    return m
end

end # module
