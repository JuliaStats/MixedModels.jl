"""
VarCorr

An encapsulation of information on the fitted random-effects
variance-covariance matrices.

# Members
* `σ`: a `Vector{Vector{T}}` of unscaled standard deviations
* `ρ`: a `Vector{Matrix{T}}` of correlation matrices
* `fnms`: a `Vector{Symbol}` of grouping factor names
* `cnms`: a `Vector{Vector{String}}` of column names
* `s`: the estimate of σ, the standard deviation of the per-observation noise.  When there is no scaling factor this value is `NaN`

The main purpose of defining this type is to isolate the logic in the show method.
"""
struct VarCorr
    σρ::NamedTuple
    s
end
VarCorr(m::MixedModel) =
   VarCorr(σρs(m), dispersion_parameter(m) ? dispersion(m) : nothing)

function Base.show(io::IO, vc::VarCorr)
    σρ = vc.σρ
    nmvec = string.([keys(σρ)...])
    cnmvec = string.(foldl(vcat, [keys(sig)...] for sig in getproperty.(values(σρ), :σ)))
    σvec = vcat(collect.(values.(getproperty.(values(σρ), :σ)))...)
    if !isnothing(vc.s)
        push!(σvec, vc.s)
        push!(nmvec, "Residual")
    end
    nmwd = maximum(textwidth.(nmvec)) + 1
    cnmwd = maximum(textwidth.(cnmvec)) + 1
    nρ = maximum(length.(getproperty.(values(σρ), :ρ)))
    varvec = abs2.(σvec)
    showσvec = showoff(σvec, :plain)
    showvarvec = showoff(varvec, :plain)
    varwd = maximum(textwidth.(showvarvec)) + 1
    stdwd = maximum(textwidth.(showσvec)) + 1
    println(io, "Variance components:")
    write(io, " "^(nmwd))
    write(io, cpad("Column", cnmwd))
    write(io, cpad("Variance", varwd))
    write(io, cpad("Std.Dev.", stdwd))
    iszero(nρ) || write(io,"  Corr.")
    println(io)
    ind = 1
    for (i,v) in enumerate(values(vc.σρ))
        write(io, rpad(nmvec[i], nmwd))
        firstrow = true
        k = length(v.σ)   # number of columns in grp factor k
        ρ = v.ρ
        ρind = 0
        for j in 1:k
            !firstrow && write(io, " "^nmwd)
            write(io, rpad(cnmvec[ind], cnmwd))
            write(io, lpad(showvarvec[ind], varwd))
            write(io, lpad(showσvec[ind], stdwd))
            for l in 1:(j - 1)
                ρind += 1
                @printf(io, "%6.2f", ρ[ρind])
            end
            println(io)
            firstrow = false
            ind += 1
        end
    end
    if !isnothing(vc.s)
        write(io, rpad(last(nmvec), nmwd))
        write(io, " "^cnmwd)
        write(io, lpad(showvarvec[ind], varwd))
        write(io, lpad(showσvec[ind], stdwd))
    end
    println(io)
end
