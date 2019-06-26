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
    σ::Vector{Vector{T}}
    ρ::Vector{Matrix{T}}
    fnms::Vector{Symbol}
    cnms::Vector{Vector{String}}
    s::T
end
function VarCorr(m::MixedModel{T}) where {T}
    fnms = fname.(m.reterms)
    cnms = Vector{String}[]
    σ = Vector{T}[]
    ρ = Matrix{T}[]
    for trm in m.reterms
        σi, ρi = stddevcor(trm)
        push!(σ, σi)
        push!(ρ, ρi)
        push!(cnms, trm.cnames)
    end
    VarCorr(σ, ρ, fnms, cnms, sdest(m))
end

function Base.show(io::IO, vc::VarCorr)
        # FIXME: Do this one term at a time
    fnms = copy(vc.fnms)
    stdm = copy(vc.σ)
    cor = vc.ρ
    cnms = reduce(append!, vc.cnms, init=String[])
    if isfinite(vc.s)
        push!(fnms, :Residual)
        push!(stdm, [1.])
        rmul!(stdm, vc.s)
        push!(cnms, "")
    end
    nmwd = maximum(map(textwidth, string.(fnms))) + 1
    write(io, "Variance components:\n")
    cnmwd = max(6, maximum(map(textwidth, cnms))) + 1
    tt = vcat(stdm...)
    vars = showoff(abs2.(tt), :plain)
    stds = showoff(tt, :plain)
    varwd = 1 + max(length("Variance"), maximum(map(textwidth, vars)))
    stdwd = 1 + max(length("Std.Dev."), maximum(map(textwidth, stds)))
    write(io, " "^(2+nmwd))
    write(io, cpad("Column", cnmwd))
    write(io, cpad("Variance", varwd))
    write(io, cpad("Std.Dev.", stdwd))
    any(s -> length(s) > 1, stdm) && write(io,"  Corr.")
    println(io)
    ind = 1
    for i in 1:length(fnms)
        stdmi = stdm[i]
        write(io, ' ')
        write(io, rpad(fnms[i], nmwd))
        write(io, rpad(cnms[ind], cnmwd))
        write(io, lpad(vars[ind], varwd))
        write(io, lpad(stds[ind], stdwd))
        ind += 1
        println(io)
        for j in 2:length(stdmi)
            write(io, " "^(1 + nmwd))
            write(io, rpad(cnms[ind], cnmwd))
            write(io, lpad(vars[ind], varwd))
            write(io, lpad(stds[ind], stdwd))
            ind += 1
            for k in 1:(j-1)
                @printf(io, "%6.2f", cor[i][j, k])
            end
            println(io)
        end
    end
end
