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
    Δdf = lrt.tests.dofdiff
    Δdev = lrt.tests.deviancediff


    nr = length(lrt.formulas)
    outrows = Vector{Vector{String}}(undef, nr+1)

    outrows[1] = ["",
                    "model-dof",
                    "deviance",
                    "χ²",
                    "χ²-dof",
                    "P(>χ²)"] # colnms

    outrows[2] = [string(lrt.formulas[1]),
                  string(lrt.dof[1]),
                  string(round(Int,lrt.deviance[1])),
                    " "," ", " "]

    for i in 2:nr
        outrows[i+1] = [string(lrt.formulas[i]),
                        string(lrt.dof[i]),
                        string(round(Int,lrt.deviance[i])),
                        string(round(Int,Δdev[i-1])),
                        string(Δdf[i-1]),
                        string(StatsBase.PValue(lrt.pvalues[i-1]))]
    end

    tbl = Markdown.Table(outrows, [:l, :r, :r, :r, :r, :l])

    show(io, Markdown.MD(tbl))
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

    bnwidth = max(maximum(length, coefnames(m)), length(_dname(m)))
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

    if dispersion_parameter(m)
        print(io, "|$(rpad(_dname(m), bnwidth))|$(string(dispersion(m))[1:cowidth])")
        print(io, "|$(" "^sewidth)|$(" "^zwidth)|$(" "^pwidth)|")
        for rr in fnames(m)
            print(io,"$(" "^σcolwidth)|")
        end
        println(io)
    end

    return nothing
end


function Base.show(io::IO, ::MIME"text/markdown", s::OptSummary)
    rows = [["",""]]

    push!(rows, ["**Initialization**", ""])
    push!(rows,["Initial parameter vector", string(s.initial)])
    push!(rows,["Initial objective value", string(s.finitial)])

    push!(rows,["**Optimizer settings** ", ""])
    push!(rows,["Optimizer (from NLopt)", "`$(s.optimizer)`"])
    push!(rows,["`Lower bounds`", string(s.lowerbd)])
    push!(rows,["`ftol_rel`", string(s.ftol_rel)])
    push!(rows,["`ftol_abs`", string(s.ftol_abs)])
    push!(rows,["`xtol_rel`", string(s.xtol_rel)])
    push!(rows,["`xtol_abs`", string(s.xtol_abs)])
    push!(rows,["`initial_step`", string(s.initial_step)])
    push!(rows,["`maxfeval`", string(s.maxfeval)])

    push!(rows,["**Result**",""])
    push!(rows,["Function evaluations", string(s.feval)])
    push!(rows,["Final parameter vector", "$(round.(s.final; digits=4))"])
    push!(rows,["Final objective value", "$(round.(s.fmin; digits=4))"])
    push!(rows,["Return code", "`$(s.returnvalue)`"])

    tbl = Markdown.Table(rows, [:l, :l])
    show(io, Markdown.MD(tbl))
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
    nmwd = maximum(textwidth.(nmvec)) + 1
    cnmwd = maximum(textwidth.(cnmvec)) + 1
    nρ = maximum(length.(getproperty.(values(σρ), :ρ)))
    varvec = abs2.(σvec)
    digits = _printdigits(σvec)
    showσvec = aligncompact(σvec, digits)
    showvarvec = aligncompact(varvec, digits)
    varwd = maximum(textwidth.(showvarvec)) + 1
    stdwd = maximum(textwidth.(showσvec)) + 1
    corwd = 6
    write(io, "|", " "^(nmwd), "|")
    write(io, cpad("Column", cnmwd), "|")
    write(io, cpad("Variance", varwd), "|")
    write(io, cpad("Std.Dev.", stdwd), "|")
    iszero(nρ) || write(io, "$(cpad("Corr.", corwd))|$(repeat("$(" "^corwd)|", nρ-1))")
    println(io)
    write(io, "|:", "-"^(nmwd-1), "|:", "-"^(cnmwd-1), "|", "-"^(varwd-1), ":|", "-"^(stdwd-1), ":|")
    iszero(nρ) || write(io, "$(repeat("$("-"^(corwd-1)):|", nρ))")
    println(io)
    ind = 1
    for (i, v) in enumerate(values(vc.σρ))
        write(io, "|$(rpad(nmvec[i], nmwd))|")
        firstrow = true
        k = length(v.σ)   # number of columns in grp factor k
        ρ = v.ρ
        ρind = 0
        for j = 1:k
            !firstrow && write(io, "|", " "^nmwd, "|")
            write(io, "$(rpad(cnmvec[ind], cnmwd))|")
            write(io, "$(lpad(showvarvec[ind], varwd))|")
            write(io, "$(lpad(showσvec[ind], stdwd))|")
            for l = 1:(j-1)
                ρind += 1
                ρval = ρ[ρind]
                ρval === -0.0 ? write(io, cpad(".", corwd)) : write(io, lpad(Ryu.writefixed(ρval, 2, true), corwd))
                write(io, "|")
            end
            ρind < nρ && write(io, "      |"^(nρ - ρind) )
            println(io)
            firstrow = false
            ind += 1
        end
    end
    if !isnothing(vc.s)
        write(io, "|", rpad(last(nmvec), nmwd))
        write(io, "|", " "^cnmwd)
        write(io, "|", lpad(showvarvec[ind], varwd))
        write(io, "|", lpad(showσvec[ind], stdwd))
        write(io, "|")
        write(io, "      |"^(nρ) )
        println(io)
    end
    return nothing
end
