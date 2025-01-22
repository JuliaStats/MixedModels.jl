module MixedModelsPRIMAExt

using MixedModels: MixedModels, LinearMixedModel, objective!
using PRIMA: PRIMA

function MixedModels.prfit!(m::LinearMixedModel)
    info = PRIMA.bobyqa!(objective!(m), copy(m.optsum.initial); xl=m.optsum.lowerbd)
    m.optsum.feval = info.nf
    return m
end

end # module
