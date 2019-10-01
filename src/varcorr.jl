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
struct VarCorr{T}
    σρ::NamedTuple
    s::T
end
VarCorr(m::MixedModel) = VarCorr(σρs(m), m.σ)

function formats(vc::VarCorr)
    σρ = vc.σρ
    nmwd = max(10, maximum(textwidth.(string.(keys(σρ))))) + 1
    cnmwd = max(6, maximum(textwidth.(string.(keys.(getproperty.(values(σρ), :σ))...,))))
    σvec = [x for x in values.(getproperty.(values(σρ), :σ))...]
    varwd = 1 + max(textwidth("Variance"), maximum(textwidth.(showoff(abs2.(σvec), :plain))))
    stdwd = 1 + max(textwidth("Std.Dev."), maximum(textwidth.(showoff(σvec, :plain))))
    nmwd, cnmwd, varwd, stdwd, maximum(length.(σvec))
end

function Base.show(io::IO, vc::VarCorr)
    nmwd, cnmwd, varwd, stdwd, nρ = formats(vc)
    println(io, "Variance components:")
    write(io, " "^(2+nmwd))
    write(io, cpad("Column", cnmwd))
    write(io, cpad("Variance", varwd))
    write(io, cpad("Std.Dev.", stdwd))
    nρ > 1 && write(io,"  Corr.")
    println(io)
    for (n,v) in pairs(σρ)
        write(io, rpad(string(n), nmwd))
        firstrow = true
        for (cn, cv) in pairs(v.σ)
            !firstrow && write(io, " "^nmwd)
            write(io, rpad(string(cn), cnmwd))
            write(io, lpad(first(showoff([abs2(cv)], :plain)), varwd))
            write(io, lpad(first(showoff([cv], :plain)), stdwd))
            firstrow = false
        end
        println(io)
    end
end
#=        # FIXME: Do this one term at a time
            for k in 1:(j-1)
                @printf(io, "%6.2f", cor[i][j, k])
            end
=#
