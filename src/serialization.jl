"""
    restoreoptsum!(m::MixedModel, io::IO; atol::Real=0, rtol::Real=atol>0 ? 0 : √eps)
    restoreoptsum!(m::MixedModel, filename; atol::Real=0, rtol::Real=atol>0 ? 0 : √eps)

Read, check, and restore the `optsum` field from a JSON stream or filename.
"""
function restoreoptsum!(m::MixedModel, filename; kwargs...)
    return open(filename, "r") do io
        return restoreoptsum!(m, io; kwargs...)
    end
end

function restoreoptsum!(
    m::LinearMixedModel{T}, io::IO; atol::Real=zero(T),
    rtol::Real=atol > 0 ? zero(T) : √eps(T),
) where {T}
    dict = JSON3.read(io)
    ops = restoreoptsum!(m.optsum, dict)
    for (par, obj_at_par) in (:initial => :finitial, :final => :fmin)
        if !isapprox(
            objective(updateL!(setθ!(m, getfield(ops, par)))), getfield(ops, obj_at_par);
            rtol, atol,
        )
            throw(
                ArgumentError(
                    "model at $par does not match stored $obj_at_par within atol=$atol, rtol=$rtol"
                ),
            )
        end
    end
    return m
end

function restoreoptsum!(
    m::GeneralizedLinearMixedModel{T}, io::IO; atol::Real=zero(T),
    rtol::Real=atol > 0 ? zero(T) : √eps(T),
) where {T}
    dict = JSON3.read(io)
    ops = m.optsum

    # need to accommodate fast and slow fits
    resize!(ops.initial, length(dict.initial))
    resize!(ops.final, length(dict.final))

    theta_beta_len = length(m.θ) + length(m.β)
    if length(dict.initial) == theta_beta_len # fast=false
        if length(ops.lowerbd) == length(m.θ)
            prepend!(ops.lowerbd, fill(-Inf, length(m.β)))
        end
        setpar! = setβθ!
        varyβ = false
    else # fast=true
        setpar! = setθ!
        varyβ = true
        if length(ops.lowerbd) != length(m.θ)
            deleteat!(ops.lowerbd, 1:length(m.β))
        end
    end
    restoreoptsum!(ops, dict)
    for (par, obj_at_par) in (:initial => :finitial, :final => :fmin)
        if !isapprox(
            deviance(pirls!(setpar!(m, getfield(ops, par)), varyβ), dict.nAGQ),
            getfield(ops, obj_at_par); rtol, atol,
        )
            throw(
                ArgumentError(
                    "model at $par does not match stored $obj_at_par within atol=$atol, rtol=$rtol"
                ),
            )
        end
    end
    return m
end

function restoreoptsum!(ops::OptSummary{T}, dict::AbstractDict) where {T}
    warn_old_version = true
    allowed_missing = (
        :lowerbd,       # never saved, -Inf not allowed in JSON
        :xtol_zero_abs, # added in v4.25.0
        :ftol_zero_abs, # added in v4.25.0
        :sigma,         # added in v4.1.0
        :fitlog,        # added in v4.1.0
        :backend,       # added in v4.30.0
        :rhobeg,        # added in v4.30.0
        :rhoend,        # added in v4.30.0
    )
    nmdiff = setdiff(
        propertynames(ops),  # names in freshly created optsum
        union!(Set(keys(dict)), allowed_missing), # names in saved optsum plus those we allow to be missing
    )
    if !isempty(nmdiff)
        throw(ArgumentError(string("optsum names: ", nmdiff, " not found in io")))
    end
    if length(setdiff(allowed_missing, keys(dict))) > 1 # 1 because :lowerbd
        @debug "" setdiff(allowed_missing, keys(dict))
        warn_old_version &&
            @warn "optsum was saved with an older version of MixedModels.jl: consider resaving."
        warn_old_version = false
    end

    if any(ops.lowerbd .> dict.initial) || any(ops.lowerbd .> dict.final)
        @debug "" ops.lowerbd dict.initial dict.final
        throw(ArgumentError("initial or final parameters in io do not satisfy lowerbd"))
    end
    for fld in (:feval, :finitial, :fmin, :ftol_rel, :ftol_abs, :maxfeval, :nAGQ, :REML)
        setproperty!(ops, fld, getproperty(dict, fld))
    end
    ops.initial_step = copy(dict.initial_step)
    ops.xtol_rel = copy(dict.xtol_rel)
    copyto!(ops.initial, dict.initial)
    copyto!(ops.final, dict.final)
    ops.optimizer = Symbol(dict.optimizer)
    ops.returnvalue = Symbol(dict.returnvalue)
    # compatibility with fits saved before the introduction of various extensions
    for prop in [:xtol_zero_abs, :ftol_zero_abs]
        fallback = getproperty(ops, prop)
        setproperty!(ops, prop, get(dict, prop, fallback))
    end
    ops.sigma = get(dict, :sigma, nothing)
    fitlog = get(dict, :fitlog, nothing)
    ops.fitlog = if isnothing(fitlog)
        # compat with fits saved before fitlog
        Table(; θ=(ops.initial, ops.finitial), objective=(ops.final, ops.fmin))
    else
        _deserialize_fitlog(fitlog, T, warn_old_version)
    end
    return ops
end

# fitlog structure in MixedModels 4.x
function _deserialize_fitlog(fitlog, T, warn_old_version::Bool)
    warn_old_version &&
        @warn "optsum was saved with an older version of MixedModels.jl: consider resaving."
    warn_old_version = false
    return Table((
        (; θ=convert(Vector{T}, first(entry)),
            objective=T(last(entry))) for entry in fitlog
    ))
end

function _deserialize_fitlog(fitlog::JSON3.Array{JSON3.Object}, T, ::Bool)
    return Table((
        (; θ=convert(Vector{T}, entry.θ),
            objective=T(entry.objective)) for entry in fitlog
    ))
end

StructTypes.StructType(::Type{<:OptSummary}) = StructTypes.Mutable()
StructTypes.excludes(::Type{<:OptSummary}) = (:lowerbd,)

"""
    saveoptsum(io::IO, m::MixedModel)
    saveoptsum(filename, m::MixedModel)

Save `m.optsum` (w/o the `lowerbd` field) in JSON format to an IO stream or a file

The reason for omitting the `lowerbd` field is because it often contains `-Inf`
values that are not allowed in JSON.
"""
saveoptsum(io::IO, m::MixedModel) = JSON3.write(io, m.optsum)
function saveoptsum(filename, m::MixedModel)
    open(filename, "w") do io
        saveoptsum(io, m)
    end
end

# TODO, maybe: something nice for the MixedModelBootstrap
