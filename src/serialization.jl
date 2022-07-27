"""
    restoreoptsum!(m::LinearMixedModel, io::IO)
    restoreoptsum!(m::LinearMixedModel, filename)

Read, check, and restore the `optsum` field from a JSON stream or filename.
"""
function restoreoptsum!(m::LinearMixedModel{T}, io::IO) where {T}
    dict = JSON3.read(io)
    ops = m.optsum
    okay =
        (setdiff(propertynames(ops), keys(dict)) == [:lowerbd]) &&
        all(ops.lowerbd .≤ dict.initial) &&
        all(ops.lowerbd .≤ dict.final)
    if !okay
        throw(ArgumentError("initial or final parameters in io do not satify lowerbd"))
    end
    for fld in (:feval, :finitial, :fmin, :ftol_rel, :ftol_abs, :maxfeval, :nAGQ, :REML)
        setproperty!(ops, fld, getproperty(dict, fld))
    end
    ops.initial_step = copy(dict.initial_step)
    ops.xtol_rel = copy(dict.xtol_rel)
    copyto!(ops.initial, dict.initial)
    copyto!(ops.final, dict.final)
    for (v, f) in (:initial => :finitial, :final => :fmin)
        if !isapprox(objective(updateL!(setθ!(m, getfield(ops, v)))), getfield(ops, f))
            throw(ArgumentError("model m at $v does not give stored $f"))
        end
    end
    ops.optimizer = Symbol(dict.optimizer)
    ops.returnvalue = Symbol(dict.returnvalue)
    # provides compatibility with fits saved before the introduction of fixed sigma
    ops.sigma = get(dict, :sigma, nothing)
    fitlog = get(dict, :fitlog, nothing)
    ops.fitlog = if isnothing(fitlog)
        # compat with fits saved before fitlog
        [(ops.initial, ops.finitial, ops.final, ops.fmin)]
    else
        [(convert(Vector{T}, first(entry)), T(last(entry))) for entry in fitlog]
    end
    return m
end

function restoreoptsum!(m::LinearMixedModel, filename)
    open(filename, "r") do io
        restoreoptsum!(m, io)
    end
end

"""
    saveoptsum(io::IO, m::LinearMixedModel)
    saveoptsum(filename, m::LinearMixedModel)

Save `m.optsum` (w/o the `lowerbd` field) in JSON format to an IO stream or a file

The reason for omitting the `lowerbd` field is because it often contains `-Inf`
values that are not allowed in JSON.
"""
saveoptsum(io::IO, m::LinearMixedModel) = JSON3.write(io, m.optsum)
function saveoptsum(filename, m::LinearMixedModel)
    open(filename, "w") do io
        saveoptsum(io, m)
    end
end

# TODO: write methods for GLMM
# TODO, maybe: something nice for the MixedModelBootstrap
