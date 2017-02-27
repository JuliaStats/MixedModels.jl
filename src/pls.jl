"""
    LinearMixedModel

Linear mixed-effects model representation

# Members
* `formula`: the formula for the model
* `mf`: the model frame, its `terms` component is used for labelling fixed effects
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
    Λ::Vector
    A::Hermitian # cross-product blocks
    L::LowerTriangular
    optsum::OptSummary{T}
end

"""
    densify(S::SparseMatrix, threshold=0.3)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `full(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(S::SparseMatrixCSC, threshold::Real = 0.3)
    m, n = size(S)
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
            copy!(view(res, (rv[k] - 1) * m + (1 : m), (j - 1) * n + (1 : n)), nzs[k])
        end
    end
    res
end
densify(A::AbstractMatrix, threshold::Real = 0.3) = A

function LinearMixedModel(f, mf, trms, Λ, wts)
    n = size(trms[1], 1)
    T = eltype(trms[end])
    sqrtwts = Diagonal([sqrt(x) for x in wts])
    wttrms =  isempty(sqrtwts) ? trms :
        size(sqrtwts, 2) == n ? [sqrtwts * t for t in trms] :
        throw(DimensionMismatch("length(wts) must be 0 or length(y)"))
    nt = length(trms)
    A = Array{AbstractMatrix}(nt, nt)
    L = Array{AbstractMatrix}(nt, nt)
    for i in 1:nt, j in 1:i
        A[i, j] = densify(wttrms[i]'wttrms[j])
        L[i, j] = copy(A[i, j])
    end
    for i in 1:length(Λ)
        Lii = L[i, i]
        if isa(Lii, Diagonal) && eltype(Lii) == Matrix{T}
            L[i, i] = Diagonal(map(LowerTriangular, Lii.diag))
        end
    end
    for i in 2:nt
        Lii = L[i, i]
        if isa(Lii, Diagonal)
            for j in 1:(i - 1)     # check for fill-in
                if !isdiag(A[i, j] * A[i, j]')
                    for k in i:nt
                        L[k, i] = full(L[k, i])
                    end
                end
            end
        end
    end
    optsum = OptSummary(getθ(Λ), lowerbd(Λ), :LN_BOBYQA;
        ftol_rel = convert(T, 1.0e-12), ftol_abs = convert(T, 1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(f, mf, wttrms, trms, sqrtwts, Λ, Hermitian(HeteroBlkdMatrix(A), :L),
        LowerTriangular(HeteroBlkdMatrix(L)), optsum)
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
        nre = [nrandomeff(t) for t in trms]
        if !issorted(nre, rev = true)
            trms = trms[sortperm(nre, rev = true)]
        end
    end
    Λ = [LT(t) for t in trms]
    push!(trms, X.m)
    push!(trms, reshape(convert(Vector{T}, DataFrames.model_response(mf)), (size(X, 1), 1)))
    LinearMixedModel(f, mf, trms, Λ, convert(Vector{T}, weights))
end

function cholBlocked!{T}(m::LinearMixedModel{T})
    A = m.A.data.blocks
    Λ = m.Λ
    L = m.L.data.blocks
    nblk = size(A, 2)
    nzblk = length(Λ)
    for j in 1:nblk
        Ljj = L[j, j]
        LjjH = isa(Ljj, Diagonal) ? Ljj : Hermitian(Ljj, :L)
        LjjLT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        if (Zblkj = j ≤ nzblk)
            scaleinflate!(Ljj, A[j, j], Λ[j])
        else
            copy!(Ljj, A[j, j])
        end
        for jj in 1:(j - 1)
            rankUpdate!(-one(T), L[j, jj], LjjH)
        end
        cholUnblocked!(Ljj, Val{:L})
        for i in (j + 1):nblk
            Lij = copy!(L[i, j], A[i, j])
            if Zblkj
                A_mul_B!(Lij, Λ[j])
                i ≤ nzblk && Ac_mul_B!(Λ[i], Lij)
            end
            for jj in 1:(j - 1)
                A_mul_Bc!(-one(T), L[i, jj], L[j, jj], one(T), Lij)
            end
            A_rdiv_Bc!(Lij, LjjLT)
        end
    end
    m
end

StatsBase.coef(m::LinearMixedModel) = fixef(m)

"""
    fit!(m::LinearMixedModel[, verbose::Bool=false[, optimizer::Symbol=:LN_BOBYQA]])

Optimize the objective of a `LinearMixedModel`.

A value for `optimizer` should be the name of an `NLopt` derivative-free optimizer
allowing for box constraints.
"""
function StatsBase.fit!{T}(m::LinearMixedModel{T}, verbose::Bool=false)
    optsum = m.optsum
    lb = optsum.lowerbd
    x = optsum.final
    copy!(x, optsum.initial)
    opt = NLopt.Opt(optsum.optimizer, length(x))
    NLopt.ftol_rel!(opt, optsum.ftol_rel) # relative criterion on objective
    NLopt.ftol_abs!(opt, optsum.ftol_abs) # absolute criterion on objective
    NLopt.xtol_rel!(opt, optsum.ftol_rel) # relative criterion on parameter values
    NLopt.xtol_abs!(opt, optsum.xtol_abs) # absolute criterion on parameter values
    NLopt.lower_bounds!(opt, lb)
    feval = 0
    function obj(x, g)
        length(g) == 0 || error("gradient not defined")
        feval += 1
        val = objective(cholBlocked!(setθ!(m, x)))
        feval == 1 && (optsum.finitial = val)
        verbose && println("f_", feval, ": ", round(val, 5), " ", x)
        val
    end
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, x)
    ## check if very small parameter values that must be non-negative can be set to zero
    xmin_ = copy(xmin)
    for i in eachindex(xmin_)
        if lb[i] == zero(T) && zero(T) < xmin_[i] < T(0.001)
            xmin_[i] = zero(T)
        end
    end
    if xmin_ ≠ xmin
        if (zeroobj = obj(xmin_, T[])) ≤ (fmin + 1.e-5)
            fmin = zeroobj
            copy!(xmin, xmin_)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    cholBlocked!(setθ!(m, xmin))

    optsum.feval = feval
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :ROUNDOFF_LIMITED, :FORCED_STOP]
        warn("NLopt optimization failure: $ret")
    end
    m
end

function fitted!{T}(v::AbstractArray{T}, m::LinearMixedModel{T})
    ## FIXME: Create and use `effects(m) -> β, b` w/o calculating β twice
    trms = m.trms
    A_mul_B!(vec(v), trms[end - 1], fixef(m))
    b = ranef(m)
    for j in eachindex(b)
        unscaledre!(vec(v), trms[j], b[j])
    end
    v
end

StatsBase.fitted{T}(m::LinearMixedModel{T}) = fitted!(Array(T, (size(m.trms[end], 1),)), m)

StatsBase.residuals(m::LinearMixedModel) = model_response(m) .- fitted(m)

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
    !isfit(m) && throw(ArgumentError("Model m has not been fit"))
    Ac_ldiv_B!(feL(m), copy!(v, m.L[end, end - 1]))
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
    q = sum(size(trms[j], 2) for j in 1:k)
    n, p = size(trms[k + 1])
    n, p, q, k
end

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest(m::LinearMixedModel) = sqrtpwrss(m) / √nobs(m)

"""
    setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})

Install `v` as the θ parameters in `m`.  Changes `m.Λ` only.
"""
function setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})
    Λ = m.Λ
    if length(v) != (ntot = sum(nlower, Λ))
        throw(DimensionMismatch("length(v) = $(length(v)), should be $ntot"))
    end
    offset = 0
    for i in eachindex(Λ)
        λ = Λ[i]
        if isa(λ, UniformScaling)
            Λ[i] = UniformScaling(v[offset += 1])
        else
            nti = nlower(λ)
            setθ!(λ, view(v, offset + (1 : nti)))
            offset += nti
        end
    end
    m
end

"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This value is the contents of the `1 × 1` bottom right block of `m.L`
"""
sqrtpwrss(m::LinearMixedModel) = m.L[end, end][1]

"""
    varest(m::LinearMixedModel)

Returns the estimate of σ², the variance of the conditional distribution of Y given B.
"""
varest(m::LinearMixedModel) = pwrss(m) / nobs(m)

"""
    pwrss(m::LinearMixedModel)

The penalized residual sum-of-squares.
"""
pwrss(m::LinearMixedModel) = abs2(sqrtpwrss(m))

Base.cor(m::LinearMixedModel) = map(λ -> stddevcor(λ)[2], m.Λ)

function StatsBase.deviance(m::LinearMixedModel)
    isfit(m) || error("Model has not been fit")
    objective(m)
end

"""
    isfit(m::LinearMixedModel)

Return a `Bool` indicating if the model has been fit.
"""
isfit(m::LinearMixedModel) = m.optsum.fmin < Inf

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

Update `m.sqrtwts` from `wts` and `m.wttrms` from `m.trms`.  Recompute `m.A` and `m.L`.
"""
function reweight!{T}(m::LinearMixedModel{T}, weights::Vector{T})
    A, wttrms, trms, sqrtwts = m.A, m.wttrms, m.trms, m.sqrtwts
    if length(weights) ≠ size(sqrtwts, 2)
        throw(DimensionMismatch("length(weights) = $(length(weights)), should be $(length(d))"))
    end
    map!(sqrt, sqrtwts.diag, weights)
    for j in eachindex(trms)
        wtj = wttrms[j]
        isa(wtj, ReMat) ? copy!(wtj.z, trms[j].z) : copy!(wtj, trms[j])
        A_mul_B!(sqrtwts, wtj)
    end
    kp2 = size(A, 2)
    for j in 1 : kp2, i in j : kp2
        Ac_mul_B!(A[i, j], wttrms[i], wttrms[j])
    end
    cholBlocked!(m)
end

function Base.show(io::IO, m::LinearMixedModel)
    if !isfit(m)
        warn("Model has not been fit")
        return nothing
    end
    n, p, q, k = size(m)
    println(io, "Linear mixed model fit by maximum likelihood")
    println(io, " ", m.formula)
    oo = objective(m)
    nums = showoff([-oo/ 2, oo, aic(m), bic(m)])
    fieldwd = max(maximum(strwidth.(nums)) + 1, 11)
    for label in [" logLik", "-2 logLik", "AIC", "BIC"]
        print(io, Base.cpad(label, fieldwd))
    end
    println(io)
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
