"""
    restoreoptsum!(m::LinearMixedModel, io::IO)
    restoreoptsum!(m::LinearMixedModel, fnm::AbstractString)

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

function restoreoptsum!(m::LinearMixedModel, fnm::AbstractString)
    open(fnm, "r") do io
        restoreoptsum!(m, io)
    end
end



"""
    saveoptsum(io::IO, m::LinearMixedModel)
    saveoptsum(fnm::AbstractString, m::LinearMixedModel)

Save `m.optsum` (w/o the `lowerbd` field) in JSON format to an IO stream or a file

The reason for omitting the `lowerbd` field is because it often contains `-Inf`
values that are not allowed in JSON.
"""
saveoptsum(io::IO, m::LinearMixedModel) = JSON3.write(io, m.optsum)
function saveoptsum(fnm::AbstractString, m::LinearMixedModel)
    open(fnm, "w") do io
        saveoptsum(io, m)
    end
end

# λ::Vector{<:Union{LowerTriangular{T,Matrix{T}},Diagonal{T,Vector{T}}}}
# inds::Vector{Vector{Int}}
# lowerbd::Vector{T}
# fcnames::NamedTuple

function savefitcollection(fnm::AbstractString, bs::MixedModelFitCollection; ftype=Float32)
    # idea:
    # 1. write the fits field as an arrow table
    # - need to convert to column table
    # - need to deal with static arrays
    # - need to convert floats to ftype (maybe do this before column table to avoid allocations?)
    # 2. write fcnames as metadata ? needs to be Dict{String, String}
    # 3. write λ, inds, lowerbd, as the first row in their own columns, with the rest of the values missing
    #    and see if comrpession/runlength encoding helps (compare size of table from step 1 to size of table after 3)

end
