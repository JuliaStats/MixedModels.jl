# for this type of union, the compiler will actually generate the necessary methods
# but it's also type stable either way
_MdTypes = Union{BlockDescription, LikelihoodRatioTest, OptSummary, VarCorr, MixedModel}
Base.show(mime::MIME, x::_MdTypes) = Base.show(Base.stdout, mime, x)


function Base.show(io::IO, ::MIME"text/markdown", b::BlockDescription)
    rowwidth = max(maximum(ndigits, b.blkrows) + 1, 5)
    colwidth = max(maximum(textwidth, b.blknms) + 1, 14)
    ncols = length(b.blknms)
    print(io, "|", rpad("rows", rowwidth), "|")
    println(io, ("$(cpad(bn, colwidth))|" for bn in b.blknms)...)
    print(io, "|", rpad(":", rowwidth, "-"), "|")
    println(io, (":$("-"^(colwidth-2)):|" for _ in b.blknms)...)
    for (i, r) in enumerate(b.blkrows)
        print(io, "|$(rpad(string(r), rowwidth))|")
        for j in 1:i
            print(io, "$(rpad(b.ALtypes[i, j],colwidth))|")
        end
        i < ncols && print(io, "$(" "^colwidth)|"^(ncols-i))
        println(io)
    end
end

function Base.show(io::IO, ::MIME"text/markdown", lrt::LikelihoodRatioTest)
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

function Base.show(io::IO, ::MIME"text/markdown", m::MixedModel)
    if m.optsum.feval < 0
        @warn("Model has not been fit: results will be nonsense")
    end
    n, p, q, k = size(m)
    REML = m.optsum.REML
    nrecols = length(fnames(m))

    co = coef(m)
    se = stderror(m)
    z = co ./ se
    p = ccdf.(Chisq(1), abs2.(z))

    bnwidth = maximum(length, coefnames(m))
    fnwidth = maximum(length ∘ string, fnames(m))
    σvec = vcat(collect.(values.(values(m.σs)))...)
    σwidth = _printdigits(σvec)
    σcolwidth = max(maximum(length ∘ string, aligncompact(σvec, σwidth)),
                    fnwidth+2) + 1 # because fn's will be preceded by σ_


    co = aligncompact(co)
    se = aligncompact(se)

    cowidth = maximum(length, co)
    sewidth = maximum(length, se)

    zwidth = maximum(length ∘ string, aligncompact(round.(z; digits=2)))
    pwidth = 6 # small value formatting

    print(io,"|$(" "^bnwidth)|$(lpad("Est.",cowidth))|$(lpad("SE", sewidth))|$(lpad("z",zwidth))|$(lpad("p",pwidth))|" )
    for rr in fnames(m)
        print(io,"σ_$(rpad(rr,σcolwidth-2))|")
    end
    println(io)

    print(io,"|:", "-"^(bnwidth-1),"|", "-"^(cowidth-1),":|", "-"^(sewidth-1), ":|", "-"^(zwidth-1),":|", "-"^(pwidth-1), ":|" )
    for rr in fnames(m)
        print(io,"-"^(σcolwidth-1), ":|")
    end
    println(io)

    for (i, bname) in enumerate(coefnames(m))

        print(io, "|$(rpad(bname, bnwidth))|$(lpad(co[i],cowidth))|$(lpad(se[i], sewidth))|")
        print(io, lpad(sprint(show, StatsBase.TestStat(z[i])), zwidth))
        print(io, "|")
        print(io, rpad(sprint(show, StatsBase.PValue(p[i])), pwidth))
        print(io, "|")

        bname = Symbol(bname)

        for (j, sig) in enumerate(m.σs)
            if bname in keys(sig)
                print(io, "$(lpad(aligncompact(getproperty(sig, bname), σwidth),σcolwidth))")
            else
                print(io, " "^σcolwidth)
            end
            print(io, "|")
        end
        println(io)
    end

    dispersion_parameter(m) && println(io, "|$(rpad(_dname(m), bnwidth))|$(string(dispersion(m))[1:cowidth])||||$("|"^nrecols)")

    return nothing
end


function Base.show(io::IO, ::MIME"text/markdown", s::OptSummary)
    println(io,"| | |")
    println(io,"|-|-|")
    println(io,"|**Initialization**| |")
    println(io,"|Initial parameter vector|", s.initial,"|")
    println(io,"|Initial objective value|", s.finitial,"|")
    println(io,"|**Optimizer settings**| |")
    println(io,"|Optimizer (from NLopt)|`", s.optimizer,"`|")
    println(io,"|`Lower bounds`|", s.lowerbd,"|")
    println(io,"|`ftol_rel`|", s.ftol_rel,"|")
    println(io,"|`ftol_abs`|", s.ftol_abs,"|")
    println(io,"|`xtol_rel`|", s.xtol_rel,"|")
    println(io,"|`xtol_abs`|", s.xtol_abs,"|")
    println(io,"|`initial_step`|", s.initial_step,"|")
    println(io,"|`maxfeval`|", s.maxfeval,"|")
    println(io,"|**Result**| |")
    println(io,"|Function evaluations|", s.feval,"|")
    println(io,"|Final parameter vector|", round.(s.final; digits=4),"|")
    println(io,"|Final objective value|", round.(s.fmin; digits=4),"|")
    println(io,"|Return code|`", s.returnvalue,"`|")
end

function Base.show(io::IO, ::MIME"text/markdown", vc::VarCorr)
    digits = 2
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
        println(io)
    end
    return nothing
end
