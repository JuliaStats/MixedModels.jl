# for this type of union, the compiler will actually generate the necessary methods
# but it's also type stable either way
_MdTypes = Union{BlockDescription,LikelihoodRatioTest,OptSummary,VarCorr,MixedModel}
Base.show(mime::MIME, x::_MdTypes) = show(Base.stdout, mime, x)

Base.show(io::IO, ::MIME"text/markdown", x::_MdTypes) = show(io, Markdown.MD(_markdown(x)))
# let's not discuss why we need show above and println below,
# nor what happens if we try display instead :)
Base.show(io::IO, ::MIME"text/html", x::_MdTypes) = println(io, Markdown.html(_markdown(x)))
# print and println because Julia already adds a newline line
Base.show(io::IO, ::MIME"text/latex", x::_MdTypes) = print(io, Markdown.latex(_markdown(x)))
function Base.show(io::IO, ::MIME"text/xelatex", x::_MdTypes)
    return print(io, Markdown.latex(_markdown(x)))
end

# not sure why this escaping doesn't work automatically
# FIXME: find out a way to get the stdlib to do this
function Base.show(io::IO, ::MIME"text/html", x::OptSummary)
    out = Markdown.html(_markdown(x))
    out = replace(out, r"&#96;([^[:space:]]*)&#96;" => s"<code>\1</code>")
    out = replace(out, r"\*\*(.*?)\*\*" => s"<b>\1</b>")
    return println(io, out)
end

function Base.show(io::IO, ::MIME"text/latex", x::OptSummary)
    out = Markdown.latex(_markdown(x))
    out = replace(out, r"`([^[:space:]]*)`" => s"\\texttt{\1}")
    out = replace(out, r"\*\*(.*?)\*\*" => s"\\textbf{\1}")
    return print(io, out)
end

function Base.show(io::IO, ::MIME"text/latex", x::MixedModel)
    la = Markdown.latex(_markdown(x))
    # take advantage of subscripting
    # including preceding & prevents capturing coefficients
    la = replace(la, r"& σ\\_([[:alnum:]]*) " => s"& $\\sigma_\\text{\1}$ ")
    return print(io, la)
end

function Base.show(io::IO, ::MIME"text/latex", x::LikelihoodRatioTest)
    la = Markdown.latex(_markdown(x))
    # take advantage of subscripting
    # including preceding & prevents capturing coefficients
    la = replace(la, r"χ²" => s"$\\chi^2$")
    return print(io, la)
end

function _markdown(b::BlockDescription)
    ncols = length(b.blknms)
    align = repeat([:l], ncols + 1)
    newrow = ["rows"; [bn for bn in b.blknms]]
    rows = [newrow]

    for (i, r) in enumerate(b.blkrows)
        newrow = [string(r)]
        for j in 1:i
            push!(newrow, "$(b.ALtypes[i, j])")
        end
        i < ncols && append!(newrow, repeat([""], ncols - i))
        push!(rows, newrow)
    end

    tbl = Markdown.Table(rows, align)
    return tbl
end

function _markdown(lrt::LikelihoodRatioTest)
    Δdf = lrt.tests.dofdiff
    Δdev = lrt.tests.deviancediff

    nr = length(lrt.formulas)
    outrows = Vector{Vector{String}}(undef, nr + 1)

    outrows[1] = ["", "model-dof", "deviance", "χ²", "χ²-dof", "P(>χ²)"] # colnms

    outrows[2] = [
        string(lrt.formulas[1]),
        string(lrt.dof[1]),
        string(round(Int, lrt.deviance[1])),
        " ",
        " ",
        " ",
    ]

    for i in 2:nr
        outrows[i + 1] = [
            string(lrt.formulas[i]),
            string(lrt.dof[i]),
            string(round(Int, lrt.deviance[i])),
            string(round(Int, Δdev[i - 1])),
            string(Δdf[i - 1]),
            string(StatsBase.PValue(lrt.pvalues[i - 1])),
        ]
    end

    tbl = Markdown.Table(outrows, [:l, :r, :r, :r, :r, :l])
    return tbl
end

_dname(::GeneralizedLinearMixedModel) = "Dispersion"
_dname(::LinearMixedModel) = "Residual"

function _markdown(m::MixedModel)
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
    align = [:l, :r, :r, :r, :r]

    for rr in fnames(m)
        push!(newrow, "σ_$(rr)")
        push!(align, :r)
    end

    rows = [newrow]

    for (i, bname) in enumerate(coefnames(m))
        newrow = [
            bname,
            Ryu.writefixed(co[i], digits),
            Ryu.writefixed(se[i], digits),
            sprint(show, StatsBase.TestStat(z[i])),
            sprint(show, StatsBase.PValue(p[i])),
        ]
        bname = Symbol(bname)

        for (j, sig) in enumerate(m.σs)
            if bname in keys(sig)
                push!(newrow, Ryu.writefixed(getproperty(sig, bname), digits))
            else
                push!(newrow, " ")
            end
        end
        push!(rows, newrow)
    end

    re_without_fe = setdiff(
        mapfoldl(x -> Set(getproperty(x, :cnames)), ∪, m.reterms), coefnames(m)
    )

    for bname in re_without_fe
        newrow = [bname, "", "", "", ""]
        bname = Symbol(bname)

        for (j, sig) in enumerate(m.σs)
            if bname in keys(sig)
                push!(newrow, Ryu.writefixed(getproperty(sig, bname), digits))
            else
                push!(newrow, " ")
            end
        end
        push!(rows, newrow)
    end

    if dispersion_parameter(m)
        newrow = [_dname(m), Ryu.writefixed(dispersion(m), digits), "", "", ""]
        for rr in fnames(m)
            push!(newrow, "")
        end
        push!(rows, newrow)
    end

    tbl = Markdown.Table(rows, align)
    return tbl
end

function _markdown(s::OptSummary)
    rows = [
        ["", ""],
        ["**Initialization**", ""],
        ["Initial parameter vector", string(s.initial)],
        ["Initial objective value", string(s.finitial)],
        ["**Optimizer settings** ", ""],
        ["Optimizer (from NLopt)", "`$(s.optimizer)`"],
        ["Lower bounds", string(s.lowerbd)],
        ["`ftol_rel`", string(s.ftol_rel)],
        ["`ftol_abs`", string(s.ftol_abs)],
        ["`xtol_rel`", string(s.xtol_rel)],
        ["`xtol_abs`", string(s.xtol_abs)],
        ["`initial_step`", string(s.initial_step)],
        ["`maxfeval`", string(s.maxfeval)],
        ["`maxtime`", string(s.maxtime)],
        ["**Result**", ""],
        ["Function evaluations", string(s.feval)],
        ["Final parameter vector", "$(round.(s.final; digits=4))"],
        ["Final objective value", "$(round.(s.fmin; digits=4))"],
        ["Return code", "`$(s.returnvalue)`"],
    ]

    tbl = Markdown.Table(rows, [:l, :l])
    return tbl
end

function _markdown(vc::VarCorr)
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

    newrow = [" ", "Column", " Variance", "Std.Dev"]
    iszero(nρ) || push!(newrow, "Corr.")
    rows = [newrow]

    align = [:l, :l, :r, :r]
    iszero(nρ) || push!(align, :r)

    ind = 1
    for (i, v) in enumerate(values(vc.σρ))
        newrow = [string(nmvec[i])]
        firstrow = true
        k = length(v.σ)   # number of columns in grp factor k
        ρ = v.ρ
        ρind = 0
        for j in 1:k
            !firstrow && push!(newrow, " ")
            push!(newrow, string(cnmvec[ind]))
            push!(newrow, string(showvarvec[ind]))
            push!(newrow, string(showσvec[ind]))
            for l in 1:(j - 1)
                ρind += 1
                ρval = ρ[ρind]
                if ρval === -0.0
                    push!(newrow, ".")
                else
                    push!(newrow, Ryu.writefixed(ρval, 2, true))
                end
            end
            push!(rows, newrow)
            newrow = Vector{String}()
            firstrow = false
            ind += 1
        end
    end
    if !isnothing(vc.s)
        newrow = [string(last(nmvec)), " ", string(showvarvec[ind]), string(showσvec[ind])]
        push!(rows, newrow)
    end

    # pad out the rows to all have the same length
    rowlen = maximum(length, rows)
    for rr in rows
        append!(rr, repeat([" "], rowlen - length(rr)))
    end
    append!(align, repeat([:r], rowlen - length(align)))

    tbl = Markdown.Table(rows, align)
    return tbl
end
