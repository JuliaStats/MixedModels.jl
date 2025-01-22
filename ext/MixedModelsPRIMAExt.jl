module MixedModelsPRIMAExt

using MixedModels: MixedModels, LinearMixedModel, objective!
using PRIMA: PRIMA

function MixedModels.prfit!(m::LinearMixedModel)
    return PRIMA.bobyqa!(objective!(m), copy(m.optsum.initial); xl=m.optsum.lowerbd)
end

end # module
