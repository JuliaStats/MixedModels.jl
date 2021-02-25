# for this type of union, the compiler will actually generate the necessary methods
# but it's also type stable either way
Base.show(mime::MIME,
          x::Union{BlockDescription, VarCorr, RandomEffectsTerm, MixedModel}) = Base.show(Base.stdout, mime, x)


function Base.show(io::IO, ::MIME"text/markdown", vc::VarCorr)
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
    digits = _printdigits(σvec)
    showσvec = aligncompact(σvec, digits)
    showvarvec = aligncompact(varvec, digits)
    println(io, "Variance components:")
    write(io, "|   |Column|Variance|Std.Dev.|")
    iszero(nρ) || write(io, "Corr.|$(repeat("    |", nρ-1))")
    println(io)
    write(io, "|:--|:-----|-------:|-------:|")
    iszero(nρ) || write(io, "----:|$(repeat("----:|", nρ-1))")
    println(io)
    ind = 1
    for (i, v) in enumerate(values(vc.σρ))
        write(io, "|$(nmvec[i])|")
        firstrow = true
        k = length(v.σ)   # number of columns in grp factor k
        ρ = v.ρ
        ρind = 0
        for j = 1:k
            !firstrow && write(io, "| |")
            write(io, "$(cnmvec[ind])|")
            write(io, "$(showvarvec[ind])|")
            write(io, "$(showσvec[ind])|")
            for l = 1:(j-1)
                ρind += 1
                ρval = ρ[ρind]
                ρval === -0.0 ? write(io, "   .  ") : write(io, lpad(Ryu.writefixed(ρval, 2, true), 6))
                write(io, "|")
            end
            ρind < nρ && write(io, " |"^(nρ - ρind) )
            println(io)
            firstrow = false
            ind += 1
        end
    end
    if !isnothing(vc.s)
        write(io, "|$(last(nmvec))| |")
        write(io, "$(showvarvec[ind])|")
        write(io, "$(showσvec[ind])|")
    end
    println(io)
end
