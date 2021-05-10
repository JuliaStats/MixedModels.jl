"""
    LinearMixedModel

Linear mixed-effects model representation

## Fields

* `formula`: the formula for the model
* `allterms`: a vector of random-effects terms, the fixed-effects terms and the response
* `reterms`: a `Vector{AbstractReMat{T}}` of random-effects terms.
* `feterms`: a `Vector{FeMat{T}}` of the fixed-effects model matrix and the response
* `sqrtwts`: vector of square roots of the case weights.  Can be empty.
* `parmap` : Vector{NTuple{3,Int}} of (block, row, column) mapping of θ to λ
* `dims` : NamedTuple{(:n, :p, :nretrms),NTuple{3,Int}} of dimensions.  `p` is the rank of `X`, which may be smaller than `size(X, 2)`.
* `A`: an `nt × nt` symmetric `BlockMatrix` of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
* `L`: a `nt × nt` `BlockMatrix` - the lower Cholesky factor of `Λ'AΛ+I`
* `optsum`: an [`OptSummary`](@ref) object

## Properties

* `θ` or `theta`: the covariance parameter vector used to form λ
* `β` or `beta`: the fixed-effects coefficient vector
* `λ` or `lambda`: a vector of lower triangular matrices repeated on the diagonal blocks of `Λ`
* `σ` or `sigma`: current value of the standard deviation of the per-observation noise
* `b`: random effects on the original scale, as a vector of matrices
* `u`: random effects on the orthogonal scale, as a vector of matrices
* `lowerbd`: lower bounds on the elements of θ
* `X`: the fixed-effects model matrix
* `y`: the response vector
"""
struct LinearMixedModel{T<:AbstractFloat} <: MixedModel{T}
    formula::FormulaTerm
    allterms::Vector{Union{AbstractReMat{T}, FeMat{T}}}
    reterms::Vector{AbstractReMat{T}}
    feterms::Vector{FeMat{T}}
    sqrtwts::Vector{T}
    parmap::Vector{NTuple{3,Int}}
    dims::NamedTuple{(:n, :p, :nretrms),NTuple{3,Int}}
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
    sch = try
        schema(f, tbl, contrasts)
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Random effects grouping variables with many levels can cause out-of-memory errors.  Try manually specifying `Grouping()` contrasts for those variables."
        end
        rethrow(e)
    end
    form = apply_schema(f, sch, LinearMixedModel)
    # tbl, _ = StatsModels.missing_omit(tbl, form)

    y, Xs = modelcols(form, tbl)

    return LinearMixedModel(y, Xs, form, wts)
end

"""
    LinearMixedModel(y, Xs, form)

Private constructor for a LinearMixedModel.

To construct a model, you only need the response (`y`), already assembled
model matrices (`Xs`), schematized formula (`form`) and weights (`wts`).
Everything else in the structure can be derived from these quantities.

!!! note
    This method is internal and experimental and so may change or disappear in
    a future release without being considered a breaking change.
"""
function LinearMixedModel(y::AbstractArray,
                           Xs::Tuple, # can't be more specific here without stressing the compiler
                           form::FormulaTerm, wts = [])
    y = reshape(float(y), (:, 1)) # y as a floating-point matrix
    T = promote_type(Float64, eltype(y))  # ensure that eltype of model matrices is at least Float64
    y = convert(Matrix{T}, y)

    reterms = AbstractReMat{T}[]
    feterms = FeMat{T}[]
    for (i, x) in enumerate(Xs)
        if isa(x, AbstractReMat{T})
            push!(reterms, x)
        elseif isa(x, ReMat) # this can occur in weird situation where x is a ReMat{U}
            # avoid keeping a second copy if unweighted
            z = convert(Matrix{T}, x.z)
            wtz = x.z === x.wtz ? z : convert(Matrix{T}, x.wtz)
            S = size(z, 1)
            x = ReMat{T,S}(x.trm, x.refs, x.levels, x.cnames, z, wtz,
                           convert(LowerTriangular{Float64, Matrix{Float64}}, x.λ),
                           x.inds,
                           convert(SparseMatrixCSC{T,Int32}, x.adjA),
                           convert(Matrix{T}, x.scratch)
                           )
            push!(reterms, x)
        else
            cnames = coefnames(form.rhs[i])
            push!(feterms, FeMat(x, isa(cnames, String) ? [cnames] : collect(cnames)))
        end
    end
    push!(feterms, FeMat(y, [""]))

    return LinearMixedModel(feterms, reterms, form, wts)
end

"""
    LinearMixedModel(feterms, reterms, form, wts=[])

Private constructor for a `LinearMixedModel` given already assembled fixed and random effects.

To construct a model, you only need a vector of `FeMat`s (the fixed-effects
model matrix and response), a vector of `AbstractReMat` (the random-effects
model matrices), the formula and the weights. Everything else in the structure
can be derived from these quantities.

!!! note
    This method is internal and experimental and so may change or disappear in
    a future release without being considered a breaking change.
"""
function LinearMixedModel(feterms::Vector{FeMat{T}}, reterms::Vector{AbstractReMat{T}},
                          form::FormulaTerm, wts=[]) where T

    # detect and combine RE terms with the same grouping var
    if length(reterms) > 1
        reterms = amalgamate(reterms)
    end

    sort!(reterms, by = nranef, rev = true)
    allterms = convert(Vector{Union{AbstractReMat{T},FeMat{T}}}, vcat(reterms, feterms))
    sqrtwts = sqrt.(convert(Vector{T}, wts))
    reweight!.(allterms, Ref(sqrtwts))
    A, L = createAL(allterms)
    lbd = foldl(vcat, lowerbd(c) for c in reterms)
    θ = foldl(vcat, getθ(c) for c in reterms)
    X = first(feterms)
    optsum = OptSummary(θ, lbd, :LN_BOBYQA, ftol_rel = T(1.0e-12), ftol_abs = T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(
        form,
        allterms,
        reterms,
        feterms,
        sqrtwts,
        mkparmap(reterms),
        (n = size(X, 1), p = X.rank, nretrms = length(reterms)),
        A,
        L,
        optsum,
        )
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


_offseterr() = throw(ArgumentError("Offsets are not supported in linear models. You can simply shift the response by the offset."))

fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl;
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    verbose::Bool = false,
    REML::Bool = false,
    offset = [],
) = !isempty(offset) ? _offseterr() : fit(
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
) = !isempty(offset) ? _offseterr() : fit(
    LinearMixedModel,
    f,
    tbl,
    wts = wts,
    contrasts = contrasts,
    verbose = verbose,
    REML = REML,
)


function StatsBase.coef(m::LinearMixedModel{T}) where {T}
    piv = first(m.feterms).piv
    invpermute!(fixef!(similar(piv, T), m), piv)
end

βs(m::LinearMixedModel) = NamedTuple{(Symbol.(coefnames(m))...,)}(coef(m))

function StatsBase.coefnames(m::LinearMixedModel)
    Xtrm = first(m.feterms)
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
        ["Coef.", "Std. Error", "z", "Pr(>|z|)"],
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
    L = m.L
    s = sdest(m)
    @static if VERSION < v"1.6.1"
        spL = LowerTriangular(SparseMatrixCSC{T, Int}(sparseL(m)))
    else
        spL = LowerTriangular(sparseL(m))
    end
    nre = size(spL, 1)
    val = Array{T,3}[]
    offset = 0
    for (i, re) in enumerate(m.reterms)
        λt = s * transpose(re.λ)
        vi = size(λt, 2)
        ℓi = length(re.levels)
        vali = Array{T}(undef, (vi, vi, ℓi))
        scratch = Matrix{T}(undef, (size(spL, 1), vi))
        for b in 1:ℓi
            fill!(scratch, zero(T))
            copyto!(view(scratch, (offset + (b - 1) * vi) .+ (1:vi), :), λt)
            ldiv!(spL, scratch)
            mul!(view(vali, :, :, b), scratch', scratch)
        end
        push!(val, vali)
        offset += vi * ℓi
    end
    val
end

function _cvtbl(arr::Array{T,3}, trm) where {T}
    merge(
        NamedTuple{(fname(trm),)}((trm.levels,)),
        columntable([NamedTuple{(:σ, :ρ)}(sdcorr(view(arr, :, :, i))) for i in axes(arr, 3)]),
        )
end

"""
    condVartables(m::LinearMixedModel)

Return the conditional covariance matrices of the random effects as a `NamedTuple` of columntables
"""
function condVartables(m::MixedModel{T}) where {T}
    NamedTuple{fnames(m)}((map(_cvtbl, condVar(m), m.reterms)...,))
end

function createAL(allterms::Vector{Union{AbstractReMat{T},FeMat{T}}}) where {T}
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
                    L[Block(l, i)] = Matrix(getblock(L, l, i))
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

StatsBase.dof(m::LinearMixedModel) = m.dims.p + nθ(m) + 1

function StatsBase.dof_residual(m::LinearMixedModel)::Int
    # nobs - rank(FE) - 1 (dispersion)
    # this differs from lme4 by not including nθ
    # a better estimate would be a number somewhere between the number of
    # variance components and the number of conditional modes
    # nobs, rank FE, num conditional modes, num grouping vars
    dd = m.dims
    dd.n - dd.p - 1
end

"""
    feL(m::LinearMixedModel)

Return the lower Cholesky factor for the fixed-effects parameters, as an `LowerTriangular`
`p × p` matrix.
"""
function feL(m::LinearMixedModel)
    k = m.dims.nretrms + 1
    LowerTriangular(m.L.blocks[k, k])
end

"""
    fit!(m::LinearMixedModel[; verbose::Bool=false, REML::Bool=false])

Optimize the objective of a `LinearMixedModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
"""
function fit!(m::LinearMixedModel{T}; verbose::Bool = false, REML::Bool = false) where {T}
    optsum = m.optsum
    # this doesn't matter for LMM, but it does for GLMM, so let's be consistent
    if optsum.feval > 0
        throw(ArgumentError("This model has already been fitted. Use refit!() instead."))
    end
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
    Xtrm = first(m.feterms)
    vv = mul!(vec(v), Xtrm, fixef!(similar(Xtrm.piv, T), m))
    for (rt, bb) in zip(m.reterms, ranef(m))
        unscaledre!(vv, rt, bb)
    end
    v
end

StatsBase.fitted(m::LinearMixedModel{T}) where {T} = fitted!(Vector{T}(undef, nobs(m)), m)

"""
    fixef!(v::Vector{T}, m::MixedModel{T})

Overwrite `v` with the pivoted fixed-effects coefficients of model `m`

For full-rank models the length of `v` must be the rank of `X`.  For rank-deficient models
the length of `v` can be the rank of `X` or the number of columns of `X`.  In the latter
case the calculated coefficients are padded with -0.0 out to the number of columns.
"""
function fixef!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    Xtrm = first(m.feterms)
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
fixef(m::LinearMixedModel{T}) where {T} = fixef!(Vector{T}(undef, first(m.feterms).rank), m)

"""
    fixefnames(m::MixedModel)

Return a (permuted and truncated in the rank-deficient case) vector of coefficient names.
"""
function fixefnames(m::LinearMixedModel{T}) where {T}
    Xtrm = first(m.feterms)
    Xtrm.cnames[1:Xtrm.rank]
end

"""
    fnames(m::MixedModel)

Return the names of the grouping factors for the random-effects terms.
"""
fnames(m::MixedModel) = (fname.(m.reterms)...,)

"""
    getθ(m::LinearMixedModel)

Return the current covariance parameter vector.
"""
getθ(m::LinearMixedModel{T}) where {T} = getθ!(Vector{T}(undef, length(m.parmap)), m)

function getθ!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    pmap = m.parmap
    if length(v) ≠ length(pmap)
        throw(DimensionMismatch("length(v) = $(length(v)) ≠ length(m.parmap) = $(length(pmap))"))
    end
    reind = 1
    λ = first(m.allterms).λ
    for (k, tp) in enumerate(pmap)
        tp1 = first(tp)
        if reind ≠ tp1
            reind = tp1
            λ = m.allterms[tp1].λ
        end
        v[k] = λ[tp[2], tp[3]]
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
        updateL!(reevaluateAend!(m))
        sum(j -> sum(abs2, view(m.L, Block(l, j))), 1:(l-1))
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

function mkparmap(reterms::Vector{AbstractReMat{T}}) where {T}
    parmap = NTuple{3,Int}[]
    for (k, trm) in enumerate(reterms)
        n = LinearAlgebra.checksquare(trm.λ)
        for ind in trm.inds
            d,r = divrem(ind-1, n)
            push!(parmap, (k, r+1, d+1))
        end
    end
    parmap
end

function StatsBase.modelmatrix(m::LinearMixedModel)
    fe = first(m.feterms)
    if fe.rank == size(fe, 2)
        fe.x
    else
        fe.x[:, invperm(fe.piv)]
    end
end

nθ(m::LinearMixedModel) = length(m.parmap)

StatsBase.nobs(m::LinearMixedModel) = m.dims.n

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m`
"""
function objective(m::LinearMixedModel{T}) where {T}
    wts = m.sqrtwts
    denomdf = T(ssqdenom(m))
    val = logdet(m) + denomdf * (one(T) + log2π + log(pwrss(m) / denomdf))
    isempty(wts) ? val : val - T(2.0) * sum(log, wts)
end

StatsBase.predict(m::LinearMixedModel) = fitted(m)

Base.propertynames(m::LinearMixedModel, private::Bool = false) = (
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
    :allterms,
    :objective,
    :pvalues,
)

"""
    pwrss(m::LinearMixedModel)

The penalized, weighted residual sum-of-squares.
"""
pwrss(m::LinearMixedModel) = abs2(last(m.L))

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
    (k = length(v)) == m.dims.nretrms || throw(DimensionMismatch(""))
    L = m.L
    for j = 1:k
        mul!(
            vec(copyto!(v[j], getblock(L, length(m.allterms), j))),
            getblock(L, k + 1, j)',
            β,
            -one(T),
            one(T),
        )
    end
    for i = k:-1:1
        Lii = getblock(L, i, i)
        vi = vec(v[i])
        ldiv!(adjoint(isa(Lii, Diagonal) ? Lii : LowerTriangular(Lii)), vi)
        for j = 1:(i-1)
            mul!(vec(v[j]), getblock(L, i, j)', vi, -one(T), one(T))
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
    ranef(m::MixedModel; uscale=false)

Return, as a `Vector{Matrix{T}}`, the conditional modes of the random effects in model `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on
the original scale.

For a named variant, see [`@raneftables`](@ref).
"""
function ranef(m::LinearMixedModel{T}; uscale = false, named=nothing) where {T}
    if named !== nothing
        Base.depwarn("the `named` keyword argument is deprecated; it has no effect. Use `raneftables` instead.", :ranef)
    end
    reterms = m.reterms
    v = [Matrix{T}(undef, size(t.z, 1), nlevs(t)) for t in reterms]
    ranef!(v, m, uscale)
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
        mul!(getblock(A, nblk, j), trmn', trm)
    end
    m
end

"""
    refit!(m::LinearMixedModel[, y::Vector]; REML=m.optsum.REML)

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.
"""
function refit!(m::LinearMixedModel; REML=m.optsum.REML)
    m.optsum.feval = -1
    fit!(reevaluateAend!(m); REML=REML)
end

function refit!(m::LinearMixedModel, y; REML=m.optsum.REML)
    resp = last(m.feterms)
    length(y) == size(resp, 1) || throw(DimensionMismatch(""))
    copyto!(resp, y)
    refit!(m; REML=REML)
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
function setθ!(m::LinearMixedModel{T}, θ::Vector{T}) where {T}
    parmap, reterms = m.parmap, m.reterms
    length(θ) == length(parmap) || throw(DimensionMismatch())
    reind = 1
    λ = first(reterms).λ
    for (tv, tr) in zip(θ, parmap)
        tr1 = first(tr)
        if reind ≠ tr1
            reind = tr1
            λ = reterms[tr1].λ
        end
        λ[tr[2], tr[3]] = tv
    end
    m
end

Base.setproperty!(m::LinearMixedModel, s::Symbol, y) =
    s == :θ ? setθ!(m, y) : setfield!(m, s, y)

function Base.show(io::IO, ::MIME"text/plain", m::LinearMixedModel)
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
        nums = Ryu.writefixed.([-oo / 2, oo, aic(m), aicc(m), bic(m)], 4)
        fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
        for label in ["  logLik", "-2 logLik", "AIC", "AICc", "BIC"]
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

Base.show(io::IO, m::LinearMixedModel) = Base.show(io, MIME"text/plain"(), m)

"""
    _coord(A::AbstractMatrix)

Return the positions and values of the nonzeros in `A` as a
`NamedTuple{(:i, :j, :v), Tuple{Vector{Int32}, Vector{Int32}, Vector{Float64}}}`
"""
function _coord(A::Diagonal)
    (i = Int32.(axes(A,1)), j = Int32.(axes(A,2)), v = A.diag)
end

function _coord(A::UniformBlockDiagonal)
    dat = A.data
    r, c, k = size(dat)
    blk = repeat(r .* (0:k-1), inner=r*c)
    (
        i = Int32.(repeat(1:r, outer=c*k) .+ blk),
        j = Int32.(repeat(1:c, inner=r, outer=k) .+ blk),
        v = vec(dat)
    )
end

function _coord(A::SparseMatrixCSC{T,Int32}) where {T}
    rv = rowvals(A)
    cv = similar(rv)
    for j in axes(A, 2), k in nzrange(A, j)
        cv[k] = j
    end
    (i = rv, j = cv, v = nonzeros(A), )
end

function _coord(A::Matrix)
    m, n = size(A)
    (
        i = Int32.(repeat(axes(A, 1), outer=n)),
        j = Int32.(repeat(axes(A, 2), inner=m)),
        v = vec(A),
    )
end

"""
    sparseL(m::LinearMixedModel{T}; full::Bool=false) where {T}

Return the lower Cholesky factor `L` as a `SparseMatrix{T,Int32}`.

`full` indicates whether the parts of `L` associated with the fixed-effects and response
are to be included.
"""
function sparseL(m::LinearMixedModel{T}; full::Bool=false) where {T}
    L, reterms = m.L, m.reterms
    nt = length(reterms) + full
    rowoffset, coloffset = 0, 0
    val = (i = Int32[], j = Int32[], v = T[])
    for i in 1:nt, j in 1:i
        Lblk = L[Block(i, j)]
        cblk = _coord(Lblk)
        append!(val.i, cblk.i .+ Int32(rowoffset))
        append!(val.j, cblk.j .+ Int32(coloffset))
        append!(val.v, cblk.v)
        if i == j
            coloffset = 0
            rowoffset += size(Lblk, 1)
        else
            coloffset += size(Lblk, 2)
        end
    end
    dropzeros!(tril!(sparse(val...,)))
end


#=
"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This is the element in the lower-right of `m.L`
"""
sqrtpwrss(m::LinearMixedModel) = last(m.L)
=#

"""
    ssqdenom(m::LinearMixedModel)

Return the denominator for penalized sums-of-squares.

For MLE, this value is the number of observations. For REML, this
value is the number of observations minus the rank of the fixed-effects matrix.
The difference is analagous to the use of n or n-1 in the denominator when
calculating the variance.
"""
function ssqdenom(m::LinearMixedModel)::Int
    n = m.dims.n
    m.optsum.REML ? n - m.dims.p : n
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
    invpermute!(v, first(m.feterms).piv)
    v
end

function StatsBase.stderror(m::LinearMixedModel{T}) where {T}
    stderror!(similar(first(m.feterms).piv, T), m)
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
            mul!(getblock(A, i, j), allterms[i]', allterms[j])
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
            copyto!(getblock(L, i, j),
                    getblock(A, i, j))
        end
    end
    for (j, cj) in enumerate(m.reterms)  # pre- and post-multiply by Λ, add I to diagonal
        scaleinflate!(getblock(L, j, j), cj)
        for i = (j+1):k         # postmultiply column by Λ
            rmulΛ!(getblock(L, i, j), cj)
        end
        for jj = 1:(j-1)        # premultiply row by Λ'
            lmulΛ!(cj', getblock(L, j, jj))
        end
    end
    for j = 1:k                         # blocked Cholesky
        Ljj = getblock(L, j, j)
        for jj = 1:(j-1)
            rankUpdate!(Hermitian(Ljj, :L), getblock(L, j, jj), -one(T), one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i = (j+1):k
            Lij = getblock(L, i, j)
            for jj = 1:(j-1)
                mul!(Lij, getblock(L, i, jj), getblock(L, j, jj)', -one(T), one(T))
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
    _zerocorr!(m::LinearMixedModel[, trmnms::Vector{Symbol}])

Rewrite the random effects specification for the grouping factors in `trmnms` to zero correlation parameter.

The default for `trmnms` is all the names of random-effects terms.

A random effects term is in the zero correlation parameter configuration when the off-diagonal elements of
λ are all zero - hence there are no correlation parameters in that term being estimated.

Note that this is numerically equivalent to specifying a formula with `zerocorr` around each random effects
term, but the `formula`  fields in the resulting model will differ. In particular, `zerocorr!` will **not**
change the original `formula`'s terms to be of type of `ZeroCorr` because this would involve changing
immutable types.  This may have implications for software that manipulates the formula of a fitted model.

This is an internal function and may disappear in a future release without being considered a breaking change.
"""
function _zerocorr!(m::LinearMixedModel{T}, trmns) where {T}
    reterms = m.reterms
    for trm in reterms
        if fname(trm) in trmns
            zerocorr!(trm)
        end
    end
    newparmap = mkparmap(reterms)
    copyto!(m.parmap, newparmap)
    resize!(m.parmap, length(newparmap))
    optsum = m.optsum
    optsum.lowerbd = mapfoldl(lowerbd, vcat, reterms)
    optsum.initial = mapfoldl(getθ, vcat, reterms)
    optsum.final = copy(optsum.initial)
    optsum.xtol_abs = fill!(copy(optsum.initial), 1.0e-10)
    optsum.initial_step = T[]

    # the model is no longer fitted
    optsum.feval = -1

    m
end

_zerocorr!(m::LinearMixedModel) = _zerocorr!(m, fnames(m))
