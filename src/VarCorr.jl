"""
    VarCorr

An encapsulation of information on the fitted random-effects
variance-covariance matrices.

# Members
* `σ`: a `Vector{Vector{T}}` of unscaled standard deviations
* `ρ`: a `Vector{Matrix{T}}` of correlation matrices
* `fnms`: a `Vector{Symbol}` of grouping factor names
* `cnms`: a `Vector{Vector{String}}` of column names
* `s`: the estimate of σ, the standard deviation of the per-observation noise.  When there
is no scaling factor this value is `NaN`

The main purpose of defining this type is to isolate the logic in the show method.
"""
type VarCorr
    σ::Vector{Vector}
    ρ::Vector{Matrix}
    fnms::Vector{Symbol}
    cnms::Vector{Vector{String}}
    s
end
function VarCorr(m::MixedModel)
    LMM = lmm(m)
    Λ, trms, fnms, cnms = LMM.Λ, LMM.trms, Symbol[], Vector{String}[]
    T = eltype(Λ[1])
    σ, ρ = Vector{T}[], Matrix{T}[]
    for i in eachindex(Λ)
        σi, ρi = stddevcor(Λ[i])
        push!(σ, σi)
        push!(ρ, ρi)
        trmi = trms[i]
        push!(fnms, trmi.fnm)
        push!(cnms, trmi.cnms)
    end
    VarCorr(σ, ρ, fnms, cnms, sdest(m))
end

function Base.show(io::IO, vc::VarCorr)
    # FIXME: Do this one term at a time
    fnms = isfinite(vc.s) ? vcat(vc.fnms,"Residual") : vc.fnms
    nmwd = maximum(map(strwidth, string.(fnms))) + 1
    write(io, "Variance components:\n")
    stdm = vc.σ
    cor = vc.ρ
    cnms = reduce(vcat, vc.cnms)
    if isfinite(vc.s)
        push!(stdm, [1.])
        stdm *= vc.s
        push!(cnms, "")
    end
    cnmwd = max(6, maximum(map(strwidth, cnms))) + 1
    tt = vcat(stdm...)
    vars = showoff(abs2.(tt), :plain)
    stds = showoff(tt, :plain)
    varwd = 1 + max(length("Variance"), maximum(map(strwidth, vars)))
    stdwd = 1 + max(length("Std.Dev."), maximum(map(strwidth, stds)))
    write(io, " "^(2+nmwd))
    write(io, Base.cpad("Column", cnmwd))
    write(io, Base.cpad("Variance", varwd))
    write(io, Base.cpad("Std.Dev.", stdwd))
    any(s -> length(s) > 1, stdm) && write(io,"  Corr.")
    println(io)
    ind = 1
    for i in 1:length(fnms)
        stdmi = stdm[i]
        write(io, ' ')
        write(io, rpad(fnms[i], nmwd))
        write(io, rpad(cnms[i], cnmwd))
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
