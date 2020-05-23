"""
    LinearMixedModel

Linear mixed-effects model representation

## Fields

* `formula`: the formula for the model
* `allterms`: a vector of random-effects terms, the fixed-effects terms and the response
* `sqrtwts`: vector of square roots of the case weights.  Can be empty.
* `A`: an `nt × nt` symmetric `BlockMatrix` of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
* `L`: a `nt × nt` `BlockMatrix` - the lower Cholesky factor of `Λ'AΛ+I`
* `optsum`: an [`OptSummary`](@ref) object

## Properties

* `θ` or `theta`: the covariance parameter vector used to form λ
* `β` or `beta`: the fixed-effects coefficient vector
* `λ` or `lambda`: a vector of lower triangular matrices repeated on the diagonal blocks of `Λ`
* `σ` or `sigma`: current value of the standard deviation of the per-observation noise
* `b`: random effects on the original scale, as a vector of matrices
* `reterms`: a `Vector{ReMat{T}}` of random-effects terms.
* `feterms`: a `Vector{FeMat{T}}` of the fixed-effects model matrix and the response
* `u`: random effects on the orthogonal scale, as a vector of matrices
* `lowerbd`: lower bounds on the elements of θ
* `X`: the fixed-effects model matrix
* `y`: the response vector
"""
struct LinearMixedModel{T<:AbstractFloat} <: MixedModel{T}
    formula::FormulaTerm
    allterms::Vector{Union{ReMat{T}, FeMat{T}}}
    sqrtwts::Vector{T}
    A::BlockMatrix{T}            # cross-product blocks
    L::BlockMatrix{T}
    optsum::OptSummary{T}
end
LinearMixedModel(f::FormulaTerm, tbl; contrasts = Dict{Symbol,Any}(), wts = []) =
    LinearMixedModel(
        f::FormulaTerm,
        Tables.columntable(tbl),
        contrasts = contrasts,
        wts = wts,
    )
function LinearMixedModel(
    f::FormulaTerm,
    tbl::Tables.ColumnTable;
    contrasts = Dict{Symbol,Any}(),
    wts = [],
)
    # TODO: perform missing_omit() after apply_schema() when improved
    # missing support is in a StatsModels release
    tbl, _ = StatsModels.missing_omit(tbl, f)
    form = apply_schema(f, schema(f, tbl, contrasts), LinearMixedModel)
    # tbl, _ = StatsModels.missing_omit(tbl, form)

    y, Xs = modelcols(form, tbl)

    y = reshape(float(y), (:, 1)) # y as a floating-point matrix
    T = eltype(y)

    reterms = ReMat{T}[]
    feterms = FeMat{T}[]
    for (i, x) in enumerate(Xs)
        if isa(x, ReMat{T})
            push!(reterms, x)
        else
            cnames = coefnames(form.rhs[i])
            push!(feterms, FeMat(x, isa(cnames, String) ? [cnames] : collect(cnames)))
        end
    end
    push!(feterms, FeMat(y, [""]))

    # detect and combine RE terms with the same grouping var
    if length(reterms) > 1
        reterms = amalgamate(reterms)
    end

    sort!(reterms, by = nranef, rev = true)

    allterms = convert(Vector{Union{ReMat{T},FeMat{T}}}, vcat(reterms, feterms))
    A, L = createAL(allterms)
    lbd = foldl(vcat, lowerbd(c) for c in reterms)
    θ = foldl(vcat, getθ(c) for c in reterms)
    optsum = OptSummary(θ, lbd, :LN_BOBYQA, ftol_rel = T(1.0e-12), ftol_abs = T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(form, allterms, sqrt.(convert(Vector{T}, wts)), A, L, optsum)
end

fit(
    ::Type{LinearMixedModel},
    f::FormulaTerm,
    tbl;
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    verbose::Bool = false,
    REML::Bool = false,
) = fit(
    LinearMixedModel,
    f,
    Tables.columntable(tbl),
    wts = wts,
    contrasts = contrasts,
    verbose = verbose,
    REML = REML,
)

fit(
    ::Type{LinearMixedModel},
    f::FormulaTerm,
    tbl::Tables.ColumnTable;
    wts = wts,
    contrasts = contrasts,
    verbose = verbose,
    REML = REML,
) = fit!(
    LinearMixedModel(f, tbl, contrasts = contrasts, wts = wts),
    verbose = verbose,
    REML = REML,
)

fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl;
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    verbose::Bool = false,
    REML::Bool = false,
) = fit(
    LinearMixedModel,
    f,
    tbl,
    wts = wts,
    contrasts = contrasts,
    verbose = verbose,
    REML = REML,
)

fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl,
    d::Normal,
    l::IdentityLink;
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    verbose::Bool = false,
    REML::Bool = false,
    offset = [],
    fast::Bool = false,
    nAGQ::Integer = 1,
) = fit(
    LinearMixedModel,
    f,
    tbl,
    wts = wts,
    contrasts = contrasts,
    verbose = verbose,
    REML = REML,
)

function StatsBase.coef(m::LinearMixedModel{T}) where {T}
    piv = fetrm(m).piv
    invpermute!(fixef!(similar(piv, T), m), piv)
end

βs(m::LinearMixedModel) = NamedTuple{(Symbol.(coefnames(m))...,)}(coef(m))

function StatsBase.coefnames(m::LinearMixedModel)
    Xtrm = fetrm(m)
    invpermute!(copy(Xtrm.cnames), Xtrm.piv)
end

function StatsBase.coeftable(m::LinearMixedModel)
    co = coef(m)
    se = stderror!(similar(co), m)
    z = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    names = coefnames(m)

    CoefTable(
        hcat(co, se, z, pvalue),
        ["Estimate", "Std.Error", "z value", "P(>|z|)"],
        names,
        4, # pvalcol
        3, # teststatcol
    )
end

"""
    condVar(m::LinearMixedModel)

Return the conditional variances matrices of the random effects.

The random effects are returned by `ranef` as a vector of length `k`,
where `k` is the number of random effects terms.  The `i`th element
is a matrix of size `vᵢ × ℓᵢ`  where `vᵢ` is the size of the
vector-valued random effects for each of the `ℓᵢ` levels of the grouping
factor.  Technically those values are the modes of the conditional
distribution of the random effects given the observed data.

This function returns an array of `k` three dimensional arrays,
where the `i`th array is of size `vᵢ × vᵢ × ℓᵢ`.  These are the
diagonal blocks from the conditional variance-covariance matrix,

    s² Λ(Λ'Z'ZΛ + I)⁻¹Λ'
"""
function condVar(m::LinearMixedModel{T}) where {T}
    retrms = m.reterms
    t1 = first(retrms)
    L11 = m.L[Block(1, 1)]
    if !isone(length(retrms)) || !isa(L11, Diagonal{T,Vector{T}})
        throw(ArgumentError("code for multiple or vector-valued r.e. not yet written"))
    end
    ll = first(t1.λ)
    Ld = L11.diag
    Array{T,3}[reshape(abs2.(ll ./ Ld) .* varest(m), (1, 1, length(Ld)))]
end

function createAL(allterms::Vector{Union{ReMat{T},FeMat{T}}}) where {T}
    k = length(allterms)
    sz = [isa(t, ReMat) ? size(t, 2) : rank(t) for t in allterms]
    A = BlockArray(undef_blocks, AbstractMatrix{T}, sz, sz)
    L = BlockArray(undef_blocks, AbstractMatrix{T}, sz, sz)
    for j = 1:k
        for i = j:k
            Lij = L[Block(i, j)] = densify(allterms[i]' * allterms[j])
            A[Block(i, j)] = deepcopy(isa(Lij, BlockedSparse) ? Lij.cscmat : Lij)
        end
    end
    nretrm = sum(Base.Fix2(isa, ReMat), allterms)
    for i = 2:nretrm      # check for fill-in due to non-nested grouping factors
        ci = allterms[i]
        for j = 1:(i-1)
            cj = allterms[j]
            if !isnested(cj, ci)
                for l = i:k
                    L[Block(l, i)] = Matrix(L[Block(l, i)])
                end
                break
            end
        end
    end
    A, L
end

StatsBase.deviance(m::LinearMixedModel) = objective(m)

GLM.dispersion(m::LinearMixedModel, sqr::Bool = false) = sqr ? varest(m) : sdest(m)

GLM.dispersion_parameter(m::LinearMixedModel) = true

StatsBase.dof(m::LinearMixedModel) = size(m)[2] + nθ(m) + 1

function StatsBase.dof_residual(m::LinearMixedModel)::Int
    # nobs - rank(FE) - 1 (dispersion)
    # this differs from lme4 by not including nθ
    # a better estimate would be a number somewhere between the number of
    # variance components and the number of conditional modes
    # nobs, rank FE, num conditional modes, num grouping vars
    nobs(m) - size(m)[2] - 1
end

"""
    feind(m::LinearMixedModel)

An internal utility to return the index in `m.allterms` of the fixed-effects term.
"""
feind(m::LinearMixedModel) = findfirst(Base.Fix2(isa, FeMat), m.allterms)

"""
    feL(m::LinearMixedModel)

Return the lower Cholesky factor for the fixed-effects parameters, as an `LowerTriangular`
`p × p` matrix.
"""
function feL(m::LinearMixedModel)
    k = feind(m)
    LowerTriangular(m.L.blocks[k, k])
end

"""
    fetrm(m::LinearMixedModel)

Return the fixed-effects term from `m.allterms`
"""
fetrm(m::LinearMixedModel) = m.allterms[feind(m)]

"""
    fit!(m::LinearMixedModel[; verbose::Bool=false, REML::Bool=false])

Optimize the objective of a `LinearMixedModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
"""
function fit!(m::LinearMixedModel{T}; verbose::Bool = false, REML::Bool = false) where {T}
    optsum = m.optsum
    opt = Opt(optsum)
    optsum.REML = REML
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = objective(updateL!(setθ!(m, x)))
        verbose && println(round(val, digits = 5), " ", x)
        val
    end
    NLopt.min_objective!(opt, obj)
    optsum.finitial = obj(optsum.initial, T[])
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    ## check if small non-negative parameter values can be set to zero
    xmin_ = copy(xmin)
    lb = optsum.lowerbd
    for i in eachindex(xmin_)
        if iszero(lb[i]) && zero(T) < xmin_[i] < T(0.001)
            xmin_[i] = zero(T)
        end
    end
    if xmin_ ≠ xmin
        if (zeroobj = obj(xmin_, T[])) ≤ (fmin + 1.e-5)
            fmin = zeroobj
            copyto!(xmin, xmin_)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    updateL!(setθ!(m, xmin))

    optsum.feval = opt.numevals
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    ret == :ROUNDOFF_LIMITED && @warn("NLopt was roundoff limited")
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end

function fitted!(v::AbstractArray{T}, m::LinearMixedModel{T}) where {T}
    ## FIXME: Create and use `effects(m) -> β, b` w/o calculating β twice
    Xtrm = fetrm(m)
    vv = mul!(vec(v), Xtrm, fixef!(similar(Xtrm.piv, T), m))
    for (rt, bb) in zip(m.reterms, ranef(m))
        unscaledre!(vv, rt, bb)
    end
    v
end

StatsBase.fitted(m::LinearMixedModel{T}) where {T} = fitted!(Vector{T}(undef, nobs(m)), m)

"""
    fixef!(v::Vector{T}, m::LinearMixedModel{T})

Overwrite `v` with the pivoted fixed-effects coefficients of model `m`

For full-rank models the length of `v` must be the rank of `X`.  For rank-deficient models
the length of `v` can be the rank of `X` or the number of columns of `X`.  In the latter
case the calculated coefficients are padded with -0.0 out to the number of columns.
"""
function fixef!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    Xtrm = fetrm(m)
    if isfullrank(Xtrm)
        ldiv!(feL(m)', copyto!(v, m.L.blocks[end, end-1]))
    else
        ldiv!(
            feL(m)',
            view(copyto!(fill!(v, -zero(T)), m.L.blocks[end, end-1]), 1:(Xtrm.rank)),
        )
    end
    v
end

"""
    fixef(m::MixedModel)

Return the fixed-effects parameter vector estimate of `m`.

In the rank-deficient case the truncated parameter vector, of length `rank(m)` is returned.
This is unlike `coef` which always returns a vector whose length matches the number of
columns in `X`.
"""
fixef(m::LinearMixedModel{T}) where {T} = fixef!(Vector{T}(undef, fetrm(m).rank), m)

"""
    fixefnames(m::MixedModel)

Return a (permuted and truncated in the rank-deficient case) vector of coefficient names. 
"""
function fixefnames(m::LinearMixedModel{T}) where {T}
    Xtrm = fetrm(m)
    Xtrm.cnames[1:Xtrm.rank]
end

"""
    fnames(m::MixedModel)

Return the names of the grouping factors for the random-effects terms.
"""
fnames(m::MixedModel) = ((tr.trm.sym for tr in m.reterms)...,)

"""
    getθ(m::LinearMixedModel)

Return the current covariance parameter vector.
"""
getθ(m::LinearMixedModel{T}) where {T} = foldl(vcat, getθ.(m.reterms))

function getθ!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    k = 0
    for t in m.reterms
        nt = nθ(t)
        getθ!(view(v, (k+1):(k+nt)), t)
        k += nt
    end
    v
end

function Base.getproperty(m::LinearMixedModel{T}, s::Symbol) where {T}
    if s == :θ || s == :theta
        getθ(m)
    elseif s == :β || s == :beta
        coef(m)
    elseif s == :βs || s == :betas
        βs(m)
    elseif s == :λ || s == :lambda
        getproperty.(m.reterms, :λ)
    elseif s == :σ || s == :sigma
        sdest(m)
    elseif s == :σs || s == :sigmas
        σs(m)
    elseif s == :σρs || s == :sigmarhos
        σρs(m)
    elseif s == :b
        ranef(m)
    elseif s == :feterms
        convert(Vector{FeMat{T}}, filter(Base.Fix2(isa, FeMat), getfield(m, :allterms)))
    elseif s == :objective
        objective(m)
    elseif s == :corr
        vcov(m, corr=true)
    elseif s == :vcov
        vcov(m, corr=false)
    elseif s == :PCA
        NamedTuple{fnames(m)}(PCA.(m.reterms))
    elseif s == :pvalues
        ccdf.(Chisq(1), abs2.(coef(m) ./ stderror(m)))
    elseif s == :reterms
        convert(Vector{ReMat{T}}, filter(Base.Fix2(isa, ReMat), getfield(m, :allterms)))
    elseif s == :stderror
        stderror(m)
    elseif s == :u
        ranef(m, uscale = true)
    elseif s == :lowerbd
        m.optsum.lowerbd
    elseif s == :X
        modelmatrix(m)
    elseif s == :y
        vec(last(m.feterms).x)
    elseif s == :rePCA
        rePCA(m)
    else
        getfield(m, s)
    end
end

"""
    issingular(m::LinearMixedModel, θ=m.θ)

Test whether the model `m` is singular if the parameter vector is `θ`.

Equality comparisons are used b/c small non-negative θ values are replaced by 0 in `fit!`.
"""
issingular(m::LinearMixedModel, θ=m.θ) = any(lowerbd(m) .== θ)

function StatsBase.leverage(m::LinearMixedModel{T}) where {T}
    # This can be done more efficiently but reusing existing tools is easier.
    # The i'th leverage value is obtained by replacing the response with the i'th
    # basis vector, updating A and L, then taking the sum of squared values of the
    # last row of L, excluding the last position.
    yorig = copy(m.y)
    l = length(m.allterms)
    value = map(eachindex(yorig)) do i
        fill!(m.y, zero(T))
        m.y[i] = one(T)
        reevaluateAend!(m)
        updateL!(m)
        sum(j -> sum(abs2, m.L[Block(l, j)]), 1:(l-1))
    end
    copyto!(m.y, yorig)
    updateL!(reevaluateAend!(m))
    value
end

function StatsBase.loglikelihood(m::LinearMixedModel)
    if m.optsum.REML
        throw(ArgumentError("loglikelihood not available for models fit by REML"))
    end
    -objective(m) / 2
end

lowerbd(m::LinearMixedModel) = m.optsum.lowerbd

function StatsBase.modelmatrix(m::LinearMixedModel)
    fe = fetrm(m)
    if fe.rank == size(fe, 2)
        fe.x
    else
        fe.x[:, invperm(fe.piv)]
    end
end

nθ(m::LinearMixedModel) = sum(nθ, m.allterms)

StatsBase.nobs(m::LinearMixedModel) = first(size(m))

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m`
"""
function objective(m::LinearMixedModel)
    wts = m.sqrtwts
    logdet(m) + ssqdenom(m) * (1 + log2π + log(varest(m))) -
    (isempty(wts) ? 0 : 2 * sum(log, wts))
end

StatsBase.predict(m::LinearMixedModel) = fitted(m)

Base.propertynames(m::LinearMixedModel, private = false) = (
    :formula,
    :sqrtwts,
    :A,
    :L,
    :optsum,
    :θ,
    :theta,
    :β,
    :beta,
    :λ,
    :lambda,
    :stderror,
    :σ,
    :sigma,
    :σs,
    :sigmas,
    :b,
    :u,
    :lowerbd,
    :X,
    :y,
    :corr,
    :vcov,
    :PCA,
    :rePCA,
    :reterms,
    :feterms,
    :objective,
    :pvalues,
)

"""
    pwrss(m::LinearMixedModel)

The penalized, weighted residual sum-of-squares.
"""
pwrss(m::LinearMixedModel) = abs2(sqrtpwrss(m))

"""
    ranef!(v::Vector{Matrix{T}}, m::MixedModel{T}, β, uscale::Bool) where {T}

Overwrite `v` with the conditional modes of the random effects for `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise
on the original scale
"""
function ranef!(
    v::Vector,
    m::LinearMixedModel{T},
    β::AbstractArray{T},
    uscale::Bool,
) where {T}
    (k = length(v)) == length(m.reterms) || throw(DimensionMismatch(""))
    L = m.L
    for j = 1:k
        mul!(
            vec(copyto!(v[j], L[Block(length(m.allterms), j)])),
            L[Block(k + 1, j)]',
            β,
            -one(T),
            one(T),
        )
    end
    for i = k:-1:1
        Lii = L[Block(i, i)]
        vi = vec(v[i])
        ldiv!(adjoint(isa(Lii, Diagonal) ? Lii : LowerTriangular(Lii)), vi)
        for j = 1:(i-1)
            mul!(vec(v[j]), L[Block(i, j)]', vi, -one(T), one(T))
        end
    end
    if !uscale
        for (t, vv) in zip(m.reterms, v)
            lmul!(t.λ, vv)
        end
    end
    v
end

ranef!(v::Vector, m::LinearMixedModel, uscale::Bool) = ranef!(v, m, fixef(m), uscale)

"""
    ranef(m::LinearMixedModel; uscale=false, named=true)

Return, as a `Vector{Vector{T}}` (`Vector{NamedVector{T}}` if `named=true`),
the conditional modes of the random effects in model `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on
the original scale.
"""
function ranef(m::LinearMixedModel{T}; uscale = false, named = false) where {T}
    reterms = m.reterms
    v = [Matrix{T}(undef, size(t.z, 1), nlevs(t)) for t in reterms]
    ranef!(v, m, uscale)
    named || return v
    vnmd = map(NamedArray, v)
    for (trm, vnm) in zip(reterms, vnmd)
        setnames!(vnm, trm.cnames, 1)
        setnames!(vnm, string.(trm.trm.contrasts.levels), 2)
    end
    vnmd
end

LinearAlgebra.rank(m::LinearMixedModel) = first(m.feterms).rank

"""
    rePCA(m::LinearMixedModel; corr::Bool=true)

Return a named tuple of the normalized cumulative variance of a principal components
analysis of the random effects covariance matrices or correlation
matrices when `corr` is `true`.

The normalized cumulative variance is the proportion of the variance for the first
principal component, the first two principal components, etc.  The last element is
always 1.0 representing the complete proportion of the variance.
"""
function rePCA(m::LinearMixedModel; corr::Bool=true)
    pca = PCA.(m.reterms, corr=corr)
    NamedTuple{fnames(m)}(getproperty.(pca,:cumvar))
end

"""
    PCA(m::LinearMixedModel; corr::Bool=true)

Return a named tuple of the principal components analysis of the random effects
covariance matrices or correlation matrices when `corr` is `true`.
"""

function PCA(m::LinearMixedModel; corr::Bool=true)
    NamedTuple{fnames(m)}(PCA.(m.reterms, corr=corr))
end

"""
    reevaluateAend!(m::LinearMixedModel)

Reevaluate the last column of `m.A` from `m.feterms`.  This function should be called
after updating the response, `m.feterms[end]`.
"""
function reevaluateAend!(m::LinearMixedModel)
    A = m.A
    trmn = reweight!(last(m.allterms), m.sqrtwts)
    nblk = length(m.allterms)
    for (j, trm) in enumerate(m.allterms)
        mul!(A[Block(nblk, j)], trmn', trm)
    end
    m
end

"""
    refit!(m::LinearMixedModel[, y::Vector])

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.
"""
refit!(m::LinearMixedModel) = fit!(reevaluateAend!(m))

function refit!(m::LinearMixedModel, y)
    resp = last(m.feterms)
    length(y) == size(resp, 1) || throw(DimensionMismatch(""))
    copyto!(resp, y)
    refit!(m)
end

StatsBase.residuals(m::LinearMixedModel) = response(m) .- fitted(m)

StatsBase.response(m::LinearMixedModel) = vec(last(m.feterms).x)

function reweight!(m::LinearMixedModel, weights)
    sqrtwts = map!(sqrt, m.sqrtwts, weights)
    reweight!.(m.allterms, Ref(sqrtwts))
    updateA!(m)
    updateL!(m)
end

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest(m::LinearMixedModel) = √varest(m)

"""
    setθ!(m::LinearMixedModel, v)

Install `v` as the θ parameters in `m`.
"""
function setθ!(m::LinearMixedModel, v)
    offset = 0
    for trm in m.reterms
        k = nθ(trm)
        setθ!(trm, view(v, (1:k) .+ offset))
        offset += k
    end
    m
end

Base.setproperty!(m::LinearMixedModel, s::Symbol, y) =
    s == :θ ? setθ!(m, y) : setfield!(m, s, y)

function Base.show(io::IO, m::LinearMixedModel)
    if m.optsum.feval < 0
        @warn("Model has not been fit")
        return nothing
    end
    n, p, q, k = size(m)
    REML = m.optsum.REML
    println(io, "Linear mixed model fit by ", REML ? "REML" : "maximum likelihood")
    println(io, " ", m.formula)
    oo = objective(m)
    if REML
        println(io, " REML criterion at convergence: ", oo)
    else
        nums = showoff([-oo / 2, oo, aic(m), bic(m)])
        fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
        for label in [" logLik", "-2 logLik", "AIC", "BIC"]
            print(io, rpad(lpad(label, (fieldwd + textwidth(label)) >> 1), fieldwd))
        end
        println(io)
        print.(Ref(io), lpad.(nums, fieldwd))
        println(io)
    end
    println(io)

    show(io, VarCorr(m))

    print(io, " Number of obs: $n; levels of grouping factors: ")
    join(io, nlevs.(m.reterms), ", ")
    println(io)
    println(io, "\n  Fixed-effects parameters:")
    show(io, coeftable(m))
end

"""
    size(m::LinearMixedModel)

Returns the size of a mixed model as a tuple of length four:
the number of observations, the number of (non-singular) fixed-effects parameters,
the number of conditional modes (random effects), the number of grouping variables
"""
function Base.size(m::LinearMixedModel)
    n, p = size(fetrm(m))
    reterms = m.reterms
    n, p, sum(size.(reterms, 2)), length(reterms)
end

"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This value is the contents of the `1 × 1` bottom right block of `m.L`
"""
sqrtpwrss(m::LinearMixedModel) = first(m.L.blocks[end, end])

"""
    ssqdenom(m::LinearMixedModel)

Return the denominator for penalized sums-of-squares.

For MLE, this value is either the number of observation for ML. For REML, this
value is the number of observations minus the rank of the fixed-effects matrix.
The difference is analagous to the use of n or n-1 in the denominator when
calculating the variance.
"""
function ssqdenom(m::LinearMixedModel)::Int
    nobs(m) - m.optsum.REML * first(m.feterms).rank
end

"""
    std(m::MixedModel)

Return the estimated standard deviations of the random effects as a `Vector{Vector{T}}`.

FIXME: This uses an old convention of isfinite(sdest(m)).  Probably drop in favor of m.σs
"""
function Statistics.std(m::LinearMixedModel)
    rl = rowlengths.(m.reterms)
    s = sdest(m)
    isfinite(s) ? rmul!(push!(rl, [1.0]), s) : rl
end

"""
    stderror!(v::AbstractVector, m::LinearMixedModel)

Overwrite `v` with the standard errors of the fixed-effects coefficients in `m`

The length of `v` should be the total number of coefficients (i.e. `length(coef(m))`).
When the model matrix is rank-deficient the coefficients forced to `-0.0` have an
undefined (i.e. `NaN`) standard error.
"""
function stderror!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    L = feL(m)
    scr = Vector{T}(undef, size(L, 2))
    s = sdest(m)
    fill!(v, zero(T) / zero(T))  # initialize to appropriate NaN for rank-deficient case
    for i in eachindex(scr)
        fill!(scr, false)
        scr[i] = true
        v[i] = s * norm(ldiv!(L, scr))
    end
    invpermute!(v, fetrm(m).piv)
    v
end

function StatsBase.stderror(m::LinearMixedModel{T}) where {T}
    stderror!(similar(fetrm(m).piv, T), m)
end

"""
    updateA!(m::LinearMixedModel)

Update the cross-product array, `m.A`, from `m.allterms`

This is usually done after a reweight! operation.
"""
function updateA!(m::LinearMixedModel)
    allterms = m.allterms
    k = length(allterms)
    A = m.A
    for j = 1:k
        for i = j:k
            mul!(A[Block(i, j)], allterms[i]', allterms[j])
        end
    end
    m
end

"""
    updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.reterms` (used for λ only)

This is the crucial step in evaluating the objective, given a new parameter value.
"""
function updateL!(m::LinearMixedModel{T}) where {T}
    A = m.A
    L = m.L
    k = length(m.allterms)
    for j = 1:k                         # copy lower triangle of A to L
        for i = j:k
            copyto!(L[Block(i, j)], A[Block(i, j)])
        end
    end
    for (j, cj) in enumerate(m.reterms)  # pre- and post-multiply by Λ, add I to diagonal
        scaleinflate!(L[Block(j, j)], cj)
        for i = (j+1):k         # postmultiply column by Λ
            rmulΛ!(L[Block(i, j)], cj)
        end
        for jj = 1:(j-1)        # premultiply row by Λ'
            lmulΛ!(cj', L[Block(j, jj)])
        end
    end
    for j = 1:k                         # blocked Cholesky
        Ljj = L[Block(j, j)]
        LjjH = isa(Ljj, Diagonal) ? Ljj : Hermitian(Ljj, :L)
        for jj = 1:(j-1)
            rankUpdate!(LjjH, L[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i = (j+1):k
            Lij = L[Block(i, j)]
            for jj = 1:(j-1)
                mul!(Lij, L[Block(i, jj)], L[Block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LjjT')
        end
    end
    m
end

"""
    varest(m::LinearMixedModel)

Returns the estimate of σ², the variance of the conditional distribution of Y given B.
"""
varest(m::LinearMixedModel) = pwrss(m) / ssqdenom(m)

"""
    zerocorr!(m::LinearMixedModel[, trmnms::Vector{Symbol}])

Rewrite the random effects specification for the grouping factors in `trmnms` to zero correlation parameter.

The default for `trmnms` is all the names of random-effects terms.

A random effects term is in the zero correlation parameter configuration when the off-diagonal elements of
λ are all zero - hence there are no correlation parameters in that term being estimated.
"""
function zerocorr!(m::LinearMixedModel{T}, trmns) where {T}
    reterms = m.reterms
    for trm in reterms
        if fname(trm) in trmns
            zerocorr!(trm)
        end
    end
    optsum = m.optsum
    optsum.lowerbd = foldl(vcat, lowerbd(c) for c in reterms)
    optsum.initial = foldl(vcat, getθ(c) for c in reterms)
    optsum.final = copy(optsum.initial)
    optsum.xtol_abs = fill!(copy(optsum.initial), 1.0e-10)
    optsum.initial_step = T[]

    # the model is no longer fitted
    optsum.feval = -1

    m
end

zerocorr!(m::LinearMixedModel) = zerocorr!(m, fnames(m))
