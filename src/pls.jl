"""
    OptSummary

Summary of an `NLopt` optimization

# Members
* `initial`: a copy of the initial parameter values in the optimization
* `final`: a copy of the final parameter values from the optimization
* `fmin`: the final value of the objective
* `feval`: the number of function evaluations
* `optimizer`: the name of the optimizer used, as a `Symbol`
"""
type OptSummary
    initial::Vector{Float64}
    final::Vector{Float64}
    fmin::Float64
    feval::Int
    optimizer::Symbol
end
function OptSummary(initial::Vector{Float64}, optimizer::Symbol)
    OptSummary(initial, initial, Inf, -1, optimizer)
end

"""
    LinearMixedModel

Linear mixed-effects model representation

# Members
* `formula`: the formula for the model
* `mf`: the model frame, mostly used to get the `terms` component for labelling fixed effects
* `wttrms`: a length `nt` vector of weighted model matrices. The last two elements are `X` and `y`.
* `trms`: a vector of unweighted model matrices.  If `isempty(sqrtwts)` the same object as `wttrms`
* `Λ`: a length `nt - 2` vector of lower triangular matrices
* `sqrtwts`: the `Diagonal` matrix of the square roots of the case weights.  Allowed to be size 0
* `A`: an `nt × nt` symmetric matrix of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
* `R`: a `nt × nt` matrix of matrices - the upper Cholesky factor of `Λ'AΛ+I`
* `opt`: an [`OptSummary`](@ref) object
"""
type LinearMixedModel{T <: AbstractFloat} <: MixedModel
    formula::Formula
    mf::ModelFrame
    wttrms::Vector
    trms::Vector
    sqrtwts::Diagonal{T}
    Λ::Vector{LowerTriangular{T, Matrix{T}}}
    A::Matrix{Any}        # symmetric cross-product blocks (upper triangle)
    R::Matrix{Any}        # right Cholesky factor in blocks.
    opt::OptSummary
end

"""
    densify(S::SparseMatrix, threshold=0.3)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `full(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(S::SparseMatrixCSC, threshold::Real = 0.3)
    m,n = size(S)
    if m == n && isdiag(S)  # convert diagonal sparse to Diagonal
        return Diagonal(diag(S))
    end
    if nnz(S)/(*(size(S)...)) ≤ threshold # very sparse matrices left as is
        return S
    end
    if isbits(eltype(S))
        return full(S)
    end
    # densify a sparse matrix whose elements are arrays of bitstypes
    nzs = nonzeros(S)
    nz1 = nzs[1]
    T = typeof(nz1)
    if !isa(nz1, Array) || !isbits(eltype(nz1)) # branch not tested
        error("Nonzeros must be a bitstype or an array of same")
    end
    sz1 = size(nz1)
    if any(x->typeof(x) ≠ T || size(x) ≠ sz1, nzs) # branch not tested
        error("Inconsistent dimensions or types in array nonzeros")
    end
    M,N = size(S)
    m,n = size(nz1, 1), size(nz1, 2) # this construction allows for nz1 to be a vector
    res = Array(eltype(nz1), M * m, N * n)
    rv = rowvals(S)
    for j in 1:size(S,2)
        for k in nzrange(S, j)
            copy!(Compat.view(res, (rv[k] - 1) * m + (1 : m), (j - 1) * n + (1 : n)), nzs[k])
        end
    end
    res
end
densify(A::AbstractMatrix, threshold::Real = 0.3) = A

function LinearMixedModel(f, mf, trms, Λ, wts)
    n = size(trms[1], 1)
    T = eltype(trms[end])
    optsum = OptSummary(mapreduce(x -> getθ(x), vcat, Λ), :None)
    sqrtwts = Diagonal([sqrt(x) for x in wts])
    wttrms =  isempty(sqrtwts) ? trms :
        size(sqrtwts, 2) == n ? [sqrtwts * t for t in trms] :
        throw(DimensionMismatch("length(wts) must be 0 or length(y)"))
    nt = length(trms)
    A, R = Array{Any}(nt, nt), Array{Any}(nt, nt)
    for j in 1 : nt, i in 1 : j
        A[i, j] = densify(wttrms[i]'wttrms[j])
        R[i, j] = copy(A[i, j])
    end
    for j in 2 : nt
        if isa(R[j, j], Diagonal) || isa(R[j, j], HBlkDiag)
            for i in 1 : (j - 1)     # check for fill-in
                if !isdiag(A[i, j]'A[i, j])
                    for k in j : nt
                        R[j, k] = full(R[j, k])
                    end
                end
            end
        end
    end
    LinearMixedModel(f, mf, wttrms, trms, sqrtwts, Λ, A, R, optsum)
end

"""
    lmm(f::DataFrames.Formula, fr::DataFrames.DataFrame; weights = [])

Create a [`LinearMixedModel`](@ref) from `f`, which contains both fixed-effects terms
and random effects, and `fr`.

The return value is ready to be `fit!` but has not yet been fit.
"""
function lmm(f::Formula, fr::AbstractDataFrame; weights::Vector = [])
    mf = ModelFrame(f, fr)
    X = ModelMatrix(mf)
    T = eltype(X.m)                                       # process the random-effects terms
    retrms = filter(x -> Meta.isexpr(x, :call) && x.args[1] == :|, mf.terms.terms)
    if isempty(retrms)
        throw(ArgumentError("$f has no random-effects terms"))
    end
    trms = Any[remat(e, mf.df) for e in retrms]
    if length(trms) > 1
        nl = [nlevs(t) for t in trms]
        trms = trms[sortperm(nl; rev = true)]
    end
    Λ = LowerTriangular{T, Matrix{T}}[LT(t) for t in trms]
    push!(trms, X.m)
    push!(trms, reshape(convert(Vector{T}, DataFrames.model_response(mf)), (size(X, 1), 1)))
    LinearMixedModel(f, mf, trms, Λ, convert(Vector{T}, weights))
end

function cfactor!(m::LinearMixedModel)
    A, Λ, R = m.A, m.Λ, m.R
    n = size(A, 1)
    for j in 1 : n, i in 1 : j
        inject!(R[i, j], A[i, j])
    end
    for i in eachindex(m.Λ)
        for j in i : n
            tscale!(Λ[i], R[i, j])
        end
        for ii in 1:i
            tscale!(R[ii,i], Λ[i])
        end
        inflate!(R[i, i])
    end
    cfactor!(R)
    m
end

StatsBase.coef(m::LinearMixedModel) = fixef(m)

"""
    fit!(m::LinearMixedModel[, verbose::Bool=false[, optimizer::Symbol=:LN_BOBYQA]])

Optimize the objective of a `LinearMixedModel`.

A value for `optimizer` should be the name of an `NLopt` derivative-free optimizer
allowing for box constraints.
"""
function StatsBase.fit!(m::LinearMixedModel, verbose::Bool=false, optimizer::Symbol=:LN_BOBYQA)
    th = getθ(m)
    k = length(th)
    opt = NLopt.Opt(optimizer, k)
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, lowerbd(m))
    feval = 0
    function obj(x::Vector{Float64}, g::Vector{Float64})
        length(g) == 0 || error("gradient not defined")
        feval += 1
        setθ!(m, x) |> cfactor! |> objective
    end
    if verbose
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            length(g) == 0 || error("gradient not defined")
            feval += 1
            val = setθ!(m, x) |> cfactor! |> objective
            print("f_$feval: $(round(val, 5)), [")
            showcompact(x[1])
            for i in 2:length(x) print(","); showcompact(x[i]) end
            println("]")
            val
        end
        NLopt.min_objective!(opt, vobj)
    else
        NLopt.min_objective!(opt, obj)
    end
    fmin, xmin, ret = NLopt.optimize(opt, th)
    ## very small parameter values often should be set to zero
    xmin1 = copy(xmin)
    modified = false
    for i in eachindex(xmin1)
        if 0. < abs(xmin1[i]) < 0.001
            modified = true
            xmin1[i] = 0.
        end
    end
    if modified  # branch not tested
        ff = setθ!(m, xmin1) |> cfactor! |> objective
        if ff ≤ (fmin + 1.e-5)  # zero components if increase in objective is negligible
            fmin = ff
            copy!(xmin, xmin1)
        else
            setθ!(m, xmin) |> cfactor!
        end
    end
    m.opt = OptSummary(th, xmin, fmin, feval, optimizer)
    if verbose
        println(ret)
    end
    m
end

"""
    lowerbd(m::LinearMixedModel)

Return the vector of lower bounds on the covariance parameter vector `θ`
"""
lowerbd(m::LinearMixedModel) = mapreduce(lowerbd, vcat, m.Λ)

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m`
"""
objective(m::LinearMixedModel) = logdet(m) + nobs(m) * (1 + log2π + log(varest(m)))

"""
    fixef!{T}(v::Vector{T}, m::LinearMixedModel{T})

Overwrite `v` with the fixed-effects coefficients of model `m`
"""
function fixef!{T}(v::AbstractVector{T}, m::LinearMixedModel{T})
    if !isfit(m)
        throw(ArgumentError("Model m has not been fit"))
    end
    A_ldiv_B!(feR(m), copy!(v, m.R[end - 1, end]))
end

"""
    fixef(m::MixedModel)

Returns the fixed-effects parameter vector estimate.
"""
fixef{T}(m::LinearMixedModel{T}) = fixef!(Array(T, (size(m)[2],)), m)

StatsBase.dof(m::LinearMixedModel) = size(m.wttrms[end - 1], 2) + sum(A -> nlower(A), m.Λ) + 1

StatsBase.loglikelihood(m::LinearMixedModel) = -deviance(m)/2

StatsBase.nobs(m::LinearMixedModel) = Int(length(m.trms[end]))

function Base.size(m::LinearMixedModel)
    k = length(m.Λ)
    trms = m.wttrms
    q = sum([size(trms[j], 2) for j in 1:k])
    n, p = size(trms[k + 1])
    n, p, q, k
end

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest{T}(m::LinearMixedModel{T}) = T(sqrtpwrss(m)/√nobs(m))

"""
    setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})

Install `v` as the θ parameters in `m`.  Changes `m.Λ` only.
"""
function setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})
    Λ = m.Λ
    offset = 0
    for i in eachindex(Λ)
        λ = Λ[i]
        nti = nlower(λ)
        setθ!(λ, Compat.view(v, offset + (1 : nti)))
        offset += nti
    end
    if length(v) ≠ offset
        throw(DimensionMismatch("length(v) = $(length(v)), should be $offset"))
    end
    m
end

"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This value is the contents of the `1 × 1` bottom right block of `m.R`
"""
sqrtpwrss{T}(m::LinearMixedModel{T}) = T(m.R[end,end][1])

"""
    varest(m::LinearMixedModel)

Returns the estimate of σ², the variance of the conditional distribution of Y given B.
"""
varest{T}(m::LinearMixedModel{T}) = T(pwrss(m)/nobs(m))

"""
    pwrss(m::LinearMixedModel)

The penalized residual sum-of-squares.
"""
pwrss(m::LinearMixedModel) = abs2(sqrtpwrss(m))

"""
    chol2cor(L::LowerTriangular)

Return the correlation matrix (symmetric, positive definite with unit diagonal)
corresponding to `L * L'`.
"""
function chol2cor(L::LowerTriangular)
    size(L, 1) == 1 && return ones(1, 1)
    res = L * L'
    d = [inv(sqrt(res[i, i])) for i in 1:size(res, 1)]
    scale!(d, scale!(res, d))
end

Base.cor(m::LinearMixedModel) = map(chol2cor, m.Λ)

function StatsBase.deviance(m::LinearMixedModel)
    isfit(m) || error("Model has not been fit")
    objective(m)
end

"""
    isfit(m::LinearMixedModel)

Return a `Bool` indicating if the model has been fit.
"""
isfit(m::LinearMixedModel) = m.opt.fmin < Inf

"""
    lrt(mods::LinearMixedModel...)

Perform sequential likelihood ratio tests on a sequence of models.

The returned value is a `DataFrame` containing information on the likelihood ratio tests.
"""
function lrt(mods::LinearMixedModel...) # not tested
    if (nm = length(mods)) <= 1
        throw(ArgumentError("at least two models are required for a likelihood ratio test"))
    end
    m1 = mods[1]
    n = nobs(m1)
    for i in 2:nm
        if nobs(mods[i]) != n
            throw(ArgumentError("number of observations must be constant across models"))
        end
    end
    mods = mods[sortperm([dof(m)::Int for m in mods])]
    degf = Int[dof(m) for m in mods]
    dev = [deviance(m)::Float64 for m in mods]
    csqr = unshift!([(dev[i-1]-dev[i])::Float64 for i in 2:nm],NaN)
    pval = unshift!([ccdf(Chisq(degf[i]-degf[i-1]),csqr[i])::Float64 for i in 2:nm],NaN)
    DataFrame(Df = degf, Deviance = dev, Chisq=csqr,pval=pval)
end

"""
    reweight!{T}(m::LinearMixedModel{T}, wts::Vector{T})

Update `m.sqrtwts` from `wts` and `m.wttrms` from `m.trms`.  Recompute `m.A` and `m.R`.
"""
function reweight!{T}(m::LinearMixedModel{T}, weights::Vector{T})
    A, wttrms, trms, sqrtwts = m.A, m.wttrms, m.trms, m.sqrtwts
       # should be able to use map!(sqrt, sqrtwts.diag, weights) but that allocates storage in v0.4
    d = sqrtwts.diag
    if length(weights) ≠ length(d)
        throw(DimensionMismatch("length(weights) = $(length(weights)), should be $(length(d))"))
    end
    for i in eachindex(d)
        d[i] = sqrt(weights[i])
    end
    for j in eachindex(trms)
        A_mul_B!(sqrtwts, copy!(wttrms[j], trms[j]))
    end
    for j in 1 : size(A, 2), i in 1 : j
        Ac_mul_B!(A[i, j], wttrms[i], wttrms[j])
    end
    cfactor!(m)
end

function Base.show(io::IO, m::LinearMixedModel)
    if !isfit(m)
        warn("Model has not been fit")
        return nothing
    end
    n,p,q,k = size(m)
    println(io, "Linear mixed model fit by maximum likelihood")
    println(io, " ", m.formula)
    oo = objective(m)
    nums = showoff([-oo/ 2, oo, aic(m), bic(m)])
    fieldwd = max(maximum(length(nums)) + 2, 11)
    print(' ')
    for label in ["logLik", "-2 logLik", "AIC", "BIC"]
        print(io, Base.cpad(label, fieldwd))
    end
    println(io)
    print(' ')
    for num in nums
        print(io, lpad(num, fieldwd))
    end
    println(io); println(io)

    show(io,VarCorr(m))

    gl = grplevels(m)
    @printf(io," Number of obs: %d; levels of grouping factors: %d", n, gl[1])
    for l in gl[2:end] @printf(io, ", %d", l) end
    println(io)
    @printf(io,"\n  Fixed-effects parameters:\n")
    show(io,coeftable(m))
end

"""
    VarCorr

An encapsulation of information on the fitted random-effects
variance-covariance matrices.

# Members
* `Λ`: the vector of lower triangular matrices from the `MixedModel`
* `fnms`: a `Vector{ASCIIString}` of grouping factor names
* `cnms`: a `Vector{Vector{ASCIIString}}` of column names
* `s`: the estimate of σ, the standard deviation of the per-observation noise

The main purpose of defining this type is to isolate the logic in the show method.
"""
type VarCorr
    Λ::Vector
    fnms::Vector
    cnms::Vector
    s::Float64
    function VarCorr(Λ::Vector, fnms::Vector, cnms::Vector, s::Number)
        length(fnms) == length(cnms) == length(Λ) || throw(DimensionMismatch(""))
        if isfinite(s) && s < 0
            error("s must be non-negative")
        end
        new(Λ, fnms, cnms, s)
    end
end
function VarCorr(m::LinearMixedModel)
    Λ, trms = m.Λ, m.trms
    VarCorr(Λ, [string(trms[i].fnm) for i in eachindex(Λ)],
        [trms[i].cnms for i in eachindex(Λ)], sdest(m))
end

function Base.show(io::IO, vc::VarCorr)
    fnms = isfinite(vc.s) ? vcat(vc.fnms,"Residual") : vc.fnms
    nmwd = maximum(map(strwidth, fnms)) + 1
    write(io, "Variance components:\n")
    stdm = [rowlengths(λ) for λ in vc.Λ]
    cnms = vcat(vc.cnms...)
    if isfinite(vc.s)
        push!(stdm, [1.])
        stdm *= vc.s
        push!(cnms, "")
    end
    cnmwd = max(6, maximum(map(strwidth, cnms))) + 1
    tt = vcat(stdm...)
    vars = showoff(abs2(tt), :plain)
    stds = showoff(tt, :plain)
    varwd = 1 + max(length("Variance"), maximum(map(strwidth, vars)))
    stdwd = 1 + max(length("Std.Dev."), maximum(map(strwidth, stds)))
    write(io, " "^(2+nmwd))
    write(io, Base.cpad("Column", cnmwd))
    write(io, Base.cpad("Variance", varwd))
    write(io, Base.cpad("Std.Dev.", stdwd))
    any(s -> length(s) > 1, stdm) && write(io,"  Corr.")
    println(io)
    cor = [chol2cor(λ) for λ in vc.Λ]
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
