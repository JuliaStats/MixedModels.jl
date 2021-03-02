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

    digits = 4

    co = coef(m)
    se = stderror(m)
    z = co ./ se
    p = ccdf.(Chisq(1), abs2.(z))

    σvec = vcat(collect.(values.(values(m.σs)))...)
    σwidth = _printdigits(σvec)

    newrow = ["", "Est.", "SE", "z", "p"]
    align = [:l, :l, :r, :r, :r]

    for rr in fnames(m)
        push!(newrow,"σ_$(rr)")
        push!(align,:r)
    end

    rows = [newrow]

    for (i, bname) in enumerate(coefnames(m))
        newrow = [bname, Ryu.writefixed(co[i],digits), Ryu.writefixed(se[i],digits),
                  sprint(show, StatsBase.TestStat(z[i])), sprint(show, StatsBase.PValue(p[i]))]
        bname = Symbol(bname)

        for (j, sig) in enumerate(m.σs)
            if bname in keys(sig)
                push!(newrow, Ryu.writefixed(getproperty(sig, bname),digits))
            else
                push!(newrow, " ")
            end
        end
        push!(rows, newrow)
    end

    if dispersion_parameter(m)
        newrow = [_dname(m), Ryu.writefixed(dispersion(m),digits), "", "", ""]
        for rr in fnames(m)
            push!(newrow, "")
        end
        push!(rows, newrow)
    end

    tbl = Markdown.Table(rows, align)
    show(io, Markdown.MD(tbl))
end


function Base.show(io::IO, ::MIME"text/markdown", s::OptSummary)
    rows = [["",""]]

    push!(rows, ["**Initialization**", ""])
    push!(rows, ["Initial parameter vector", string(s.initial)])
    push!(rows, ["Initial objective value", string(s.finitial)])

    push!(rows, ["**Optimizer settings** ", ""])
    push!(rows, ["Optimizer (from NLopt)", "`$(s.optimizer)`"])
    push!(rows, ["`Lower bounds`", string(s.lowerbd)])
    push!(rows, ["`ftol_rel`", string(s.ftol_rel)])
    push!(rows, ["`ftol_abs`", string(s.ftol_abs)])
    push!(rows, ["`xtol_rel`", string(s.xtol_rel)])
    push!(rows, ["`xtol_abs`", string(s.xtol_abs)])
    push!(rows, ["`initial_step`", string(s.initial_step)])
    push!(rows, ["`maxfeval`", string(s.maxfeval)])

    push!(rows, ["**Result**",""])
    push!(rows, ["Function evaluations", string(s.feval)])
    push!(rows, ["Final parameter vector", "$(round.(s.final; digits=4))"])
    push!(rows, ["Final objective value", "$(round.(s.fmin; digits=4))"])
    push!(rows, ["Return code", "`$(s.returnvalue)`"])

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
    nρ = maximum(length.(getproperty.(values(σρ), :ρ)))
    varvec = abs2.(σvec)
    digits = _printdigits(σvec)
    showσvec = aligncompact(σvec, digits)
    showvarvec = aligncompact(varvec, digits)

    newrow = [" ", "Column"," Variance", "Std.Dev"]
    iszero(nρ) || append!(push!(newrow, "Corr."), repeat([" "], nρ-1))
    rows = [newrow]

    align = [:l, :l, :r, :r]
    iszero(nρ) || append!(align, repeat([:r], nρ))

    ind = 1
    for (i, v) in enumerate(values(vc.σρ))
        newrow = [string(nmvec[i])]
        firstrow = true
        k = length(v.σ)   # number of columns in grp factor k
        ρ = v.ρ
        ρind = 0
        for j = 1:k
            !firstrow && push!(newrow, " ")
            push!(newrow, string(cnmvec[ind]))
            push!(newrow, string(showvarvec[ind]))
            push!(newrow, string(showσvec[ind]))
            for l = 1:(j-1)
                ρind += 1
                ρval = ρ[ρind]
                ρval === -0.0 ? push!(newrow, ".") : push!(newrow, Ryu.writefixed(ρval, 2, true))
            end
            ρind < nρ && append!(newrow, repeat([" "], nρ-ρind))
            push!(rows, newrow)
            newrow = Vector{String}()
            firstrow = false
            ind += 1
        end

    end
    if !isnothing(vc.s)
        newrow = [string(last(nmvec)), " ", string(showvarvec[ind]), string(showσvec[ind])]
        append!(newrow, repeat([" "], nρ))
        push!(rows, newrow)
    end
    tbl = Markdown.Table(rows, align)
    show(io, Markdown.MD(tbl))
end
