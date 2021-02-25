# for this type of union, the compiler will actually generate the necessary methods
# but it's also type stable either way
Base.show(mime::MIME,
          x::Union{BlockDescription, LikelihoodRatioTest, VarCorr, MixedModel}) = Base.show(Base.stdout, mime, x)


function Base.show(io::IO, ::MIME"text/markdown", b::BlockDescription)
    ncols = length(b.blknms)
    print(io, "|rows|")
    println(io, ("$(bn)|" for bn in b.blknms)...)
    print(io, "|:--|")
    println(io, (":--:|" for _ in b.blknms)...)
    for (i, r) in enumerate(b.blkrows)
        print(io, "|$(string(r))|")
        for j in 1:i
            print(io, "$(b.ALtypes[i, j])|")
        end
        i < ncols && print(io, "$("|"^(ncols-i))")
        println(io)
    end
end

function Base.show(io::IO, ::MIME"text/markdown", lrt::LikelihoodRatioTest; digits=2)
    # println(io, "Model Formulae")

    # for (i, f) in enumerate(lrt.formulas)
    #     println(io, "$i: $f")
    # end

    # the following was adapted from StatsModels#162
    # from nalimilan
    Δdf = lrt.tests.dofdiff
    Δdev = lrt.tests.deviancediff

    nc = 6
    nr = length(lrt.formulas)
    outrows = Matrix{String}(undef, nr+2, nc)

    outrows[1, :] = ["",
                    "model-dof",
                    "deviance",
                    "χ²",
                    "χ²-dof",
                    "P(>χ²)"] # colnms

    outrows[2, :] = [":-", "-:", "-:",
                     "-:", "-:", ":-"]

    outrows[3, :] = ["$(replace(lrt.formulas[1], "|" => "\\|"))",
                    string(lrt.dof[1]),
                    string(round(Int,lrt.deviance[1])),
                    " "," ", " "]

    for i in 2:nr
        outrows[i+2, :] = ["$(replace(lrt.formulas[i], "|" => "\\|"))",
                           string(lrt.dof[i]),
                           string(round(Int,lrt.deviance[i])),
                           string(round(Int,Δdev[i-1])),
                           string(Δdf[i-1]),
                           string(StatsBase.PValue(lrt.pvalues[i-1]))]
    end
    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]
    totwidth = sum(max_colwidths) + 2*5

    for r in 1:nr+2
        print(io, "|")
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)

            print(io, "$(cur_cell)|")
        end
        print(io, "\n")

    end

    nothing
end



_dname(::GeneralizedLinearMixedModel) = "Dispersion"
_dname(::LinearMixedModel) = "Residual"

function Base.show(io::IO, ::MIME"text/markdown", m::MixedModel; digits=2)
    if m.optsum.feval < 0
        @warn("Model has not been fit")
        return nothing
    end
    n, p, q, k = size(m)
    REML = m.optsum.REML
    nrecols = length(fnames(m))

    print(io,"| |Est.|SE |z  |p  | " )
    for rr in fnames(m)
        print("σ_$(rr)|")
    end
    println(io)

    print(io,"|:-|----:|--:|--:|--:|" )
    for rr in fnames(m)
        print("------:|")
    end
    println(io)

    co = coef(m)
    se = stderror(m)
    z = co ./ se
    p = ccdf.(Chisq(1), abs2.(z))


    for (i, bname) in enumerate(coefnames(m))

        print(io, "|$(bname)|$(round(co[i]; digits=digits))|$(round(se[i]; digits=digits))|")
        show(io, StatsBase.TestStat(z[i]))
        print(io, "|")
        show(io, StatsBase.PValue(p[i]))
        print(io, "|")

        bname = Symbol(bname)

        for (j, sig) in enumerate(m.σs)
            bname in keys(sig) && print(io, "$(round(getproperty(sig, bname); digits=digits))")
            print(io, "|")
        end
        println()
    end

    dispersion_parameter(m) && println("|$(_dname(m))|$(round(dispersion(m); digits=digits))||||$("|"^nrecols)")

    return nothing
end


function Base.show(io::IO, ::MIME"text/markdown", vc::VarCorr)
    σρ = vc.σρ
    nmvec = string.([keys(σρ)...])
    cnmvec = string.(foldl(vcat, [keys(sig)...] for sig in getproperty.(values(σρ), :σ)))
    σvec = vcat(collect.(values.(getproperty.(values(σρ), :σ)))...)
    if !isnothing(vc.s)
        push!(σvec, vc.s)
        push!(nmvec, "Residual")
    end
    nρ = maximum(length.(getproperty.(values(σρ), :ρ)))
    varvec = abs2.(σvec)
    digits = _printdigits(σvec)
    showσvec = aligncompact(σvec, digits)
    showvarvec = aligncompact(varvec, digits)
    # println(io, "Variance components:")
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
