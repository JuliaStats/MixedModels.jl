module MixedModelsPRIMAExt

using MixedModels: MixedModels, LinearMixedModel, objective!
using PRIMA: PRIMA

function MixedModels.prfit!(m::LinearMixedModel)
    PRIMA.bobyqa!(objective!(m), copy(m.optsum.initial); xl=m.optsum.lowerbd)
    return m
end

end # module
