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
    dict = JSON.parse(io)
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
    dict = JSON.parse(io)
    ops = m.optsum

    # need to accommodate fast and slow fits
    resize!(ops.initial, length(dict["initial"]))
    resize!(ops.final, length(dict["final"]))

    theta_beta_len = length(m.θ) + length(m.β)
    if length(dict["initial"]) == theta_beta_len # fast=false
        setpar! = setβθ!
        varyβ = false
    else # fast=true
        setpar! = setθ!
        varyβ = true
    end
    restoreoptsum!(ops, dict)
    for (par, obj_at_par) in (:initial => :finitial, :final => :fmin)
        if !isapprox(
            deviance(pirls!(setpar!(m, getfield(ops, par)), varyβ), dict["nAGQ"]),
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
        :lowerbd,           # never saved, -Inf not allowed in JSON, not used in v5
        :xtol_zero_abs,     # added in v4.25.0
        :ftol_zero_abs,     # added in v4.25.0
        :sigma,             # added in v4.1.0
        :fitlog,            # added in v4.1.0
        :backend,           # added in v4.30.0
        :rhobeg,            # added in v4.30.0
        :rhoend,            # added in v4.30.0
        :pirls_maxiter,     # added in v5.6.0
        :pirls_ftol_rel,    # added in v5.6.0
        :pirls_ftol_abs,    # added in v5.6.0
        :pirls_maxhalfstep, # added in v5.6.0
    )
    dict_keys = Set(Symbol.(keys(dict)))
    nmdiff = setdiff(
        propertynames(ops),  # names in freshly created optsum
        union(dict_keys, allowed_missing), # names in saved optsum plus those we allow to be missing
    )
    if !isempty(nmdiff)
        throw(ArgumentError(string("optsum names: ", nmdiff, " not found in io")))
    end
    if length(setdiff(allowed_missing, dict_keys)) > 1 # 1 because :lowerbd
        @debug "" setdiff(allowed_missing, dict_keys)
        warn_old_version &&
            @warn "optsum was saved with an older version of MixedModels.jl: consider resaving."
        warn_old_version = false
    end

    for fld in (:feval, :finitial, :fmin, :ftol_rel, :ftol_abs, :maxfeval, :nAGQ, :REML)
        setproperty!(ops, fld, dict[string(fld)])
    end
    ops.initial_step = convert(Vector{T}, dict["initial_step"])
    ops.xtol_rel = T(dict["xtol_rel"])
    copyto!(ops.initial, dict["initial"])
    copyto!(ops.final, dict["final"])
    ops.optimizer = Symbol(dict["optimizer"])
    ops.returnvalue = Symbol(dict["returnvalue"])
    # compatibility with fits saved before the introduction of various extensions
    for prop in (
        :xtol_zero_abs,
        :ftol_zero_abs,
        :pirls_maxiter,
        :pirls_ftol_rel,
        :pirls_ftol_abs,
        :pirls_maxhalfstep,
    )
        fallback = getproperty(ops, prop)
        setproperty!(ops, prop, get(dict, string(prop), fallback))
    end
    ops.sigma = get(dict, "sigma", nothing)
    fitlog = get(dict, "fitlog", nothing)
    ops.fitlog = _deserialize_fitlog(fitlog, ops, warn_old_version)
    return ops
end

# before there was a fitlog....
function _deserialize_fitlog(::Nothing, ops::OptSummary{T}, ::Bool) where {T}
    # no need to warn here because we already warned with the missing field
    return Table(; θ=Vector{T}[ops.initial, ops.final], objective=T[ops.finitial, ops.fmin])
end

function _deserialize_fitlog(fitlog, ops::OptSummary{T}, warn_old_version::Bool) where {T}
    isempty(fitlog) &&
        return _deserialize_fitlog(nothing, ops, warn_old_version)
    if first(fitlog) isa AbstractDict
        # current format: each entry is {"θ": [...], "objective": ...}
        return Table((
            (; θ=convert(Vector{T}, entry["θ"]),
                objective=T(entry["objective"])) for entry in fitlog
        ))
    else
        # old 4.x format: each entry is [[θ...], objective]
        warn_old_version &&
            @warn "optsum was saved with an older version of MixedModels.jl: consider resaving."
        return Table((
            (; θ=convert(Vector{T}, first(entry)),
                objective=T(last(entry))) for entry in fitlog
        ))
    end
end

function _optsum_to_dict(ops::OptSummary)
    d = Dict{String,Any}()
    for fn in fieldnames(typeof(ops))
        val = getfield(ops, fn)
        if val isa Symbol
            d[string(fn)] = string(val)
        elseif val isa Table
            d[string(fn)] = [
                Dict{String,Any}("θ" => collect(row.θ), "objective" => row.objective)
                for row in Tables.rows(val)
            ]
        else
            d[string(fn)] = val
        end
    end
    return d
end

"""
    saveoptsum(io::IO, m::MixedModel)
    saveoptsum(filename, m::MixedModel)

Save `m.optsum` in JSON format to an IO stream or a file
"""
function saveoptsum(io::IO, m::MixedModel)
    JSON.print(io, _optsum_to_dict(m.optsum))
    truncate(io, position(io))
    return nothing
end
function saveoptsum(filename, m::MixedModel)
    open(filename, "w") do io
        saveoptsum(io, m)
    end
end

# TODO, maybe: something nice for the MixedModelBootstrap
