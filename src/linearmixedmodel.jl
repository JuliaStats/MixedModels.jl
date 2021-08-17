"""
    LinearMixedModel

Linear mixed-effects model representation

## Fields

* `formula`: the formula for the model
* `reterms`: a `Vector{AbstractReMat{T}}` of random-effects terms.
* `Xymat`: horizontal concatenation of a full-rank fixed-effects model matrix `X` and response `y` as an `FeMat{T}`
* `feterm`: the fixed-effects model matrix as an `FeTerm{T}`
* `sqrtwts`: vector of square roots of the case weights.  Can be empty.
* `parmap` : Vector{NTuple{3,Int}} of (block, row, column) mapping of θ to λ
* `dims` : NamedTuple{(:n, :p, :nretrms),NTuple{3,Int}} of dimensions.  `p` is the rank of `X`, which may be smaller than `size(X, 2)`.
* `A`: a `Vector{AbstractMatrix}` containing the row-major packed lower triangle of `hcat(Z,X,y)'hcat(Z,X,y)`
* `L`: the blocked lower Cholesky factor of `Λ'AΛ+I` in the same Vector representation as `A`
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
    reterms::Vector{AbstractReMat{T}}
    Xymat::FeMat{T}
    feterm::FeTerm{T}
    sqrtwts::Vector{T}
    parmap::Vector{NTuple{3,Int}}
    dims::NamedTuple{(:n, :p, :nretrms),NTuple{3,Int}}
    A::Vector{AbstractMatrix{T}}            # cross-product blocks
    L::Vector{AbstractMatrix{T}}
    optsum::OptSummary{T}
    σ::Union{T,Nothing}
end
function LinearMixedModel(f::FormulaTerm, tbl; contrasts=Dict{Symbol,Any}(), wts=[], σ=nothing)
    return LinearMixedModel(f::FormulaTerm, Tables.columntable(tbl); contrasts, wts, σ)
end

function LinearMixedModel(f::FormulaTerm, tbl::Tables.ColumnTable;
                          contrasts=Dict{Symbol,Any}(), wts=[], σ=nothing)
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

    return LinearMixedModel(y, Xs, form, wts, σ)
end

"""
    LinearMixedModel(y, Xs, form, wts=[], σ=nothing)

Private constructor for a LinearMixedModel.

To construct a model, you only need the response (`y`), already assembled
model matrices (`Xs`), schematized formula (`form`) and weights (`wts`).
Everything else in the structure can be derived from these quantities.

!!! note
    This method is internal and experimental and so may change or disappear in
    a future release without being considered a breaking change.
"""
function LinearMixedModel(y::AbstractArray, Xs::Tuple, # can't be more specific here without stressing the compiler
                          form::FormulaTerm, wts=[], σ=nothing)
    T = promote_type(Float64, float(eltype(y)))  # ensure eltype of model matrices is at least Float64

    reterms = AbstractReMat{T}[]
    feterms = FeTerm{T}[]
    for (i, x) in enumerate(Xs)
        if isa(x, AbstractReMat{T})
            push!(reterms, x)
        elseif isa(x, ReMat) # this can occur in weird situation where x is a ReMat{U}
            # avoid keeping a second copy if unweighted
            z = convert(Matrix{T}, x.z)
            wtz = x.z === x.wtz ? z : convert(Matrix{T}, x.wtz)
            S = size(z, 1)
            x = ReMat{T,S}(
                x.trm,
                x.refs,
                x.levels,
                x.cnames,
                z,
                wtz,
                convert(LowerTriangular{Float64,Matrix{Float64}}, x.λ),
                x.inds,
                convert(SparseMatrixCSC{T,Int32}, x.adjA),
                convert(Matrix{T}, x.scratch),
            )
            push!(reterms, x)
        else
            cnames = coefnames(form.rhs[i])
            push!(feterms, FeTerm(x, isa(cnames, String) ? [cnames] : collect(cnames)))
        end
    end
    return LinearMixedModel(convert(Array{T}, y), only(feterms), reterms, form, wts, σ)
end

"""
    LinearMixedModel(y, feterm, reterms, form, wts=[], σ=nothing)

Private constructor for a `LinearMixedModel` given already assembled fixed and random effects.

To construct a model, you only need a vector of `FeMat`s (the fixed-effects
model matrix and response), a vector of `AbstractReMat` (the random-effects
model matrices), the formula and the weights. Everything else in the structure
can be derived from these quantities.

!!! note
    This method is internal and experimental and so may change or disappear in
    a future release without being considered a breaking change.
"""
function LinearMixedModel(y::AbstractArray, feterm::FeTerm{T},
                          reterms::AbstractVector{<:AbstractReMat{T}},
                          form::FormulaTerm, wts=[], σ=nothing) where {T}
    # detect and combine RE terms with the same grouping var
    if length(reterms) > 1
        reterms = amalgamate(reterms)
    end

    sort!(reterms; by=nranef, rev=true)
    Xy = FeMat(feterm, vec(y))
    sqrtwts = sqrt.(convert(Vector{T}, wts))
    reweight!.(reterms, Ref(sqrtwts))
    reweight!(Xy, sqrtwts)
    A, L = createAL(reterms, Xy)
    lbd = foldl(vcat, lowerbd(c) for c in reterms)
    θ = foldl(vcat, getθ(c) for c in reterms)
    optsum = OptSummary(θ, lbd, :LN_BOBYQA; ftol_rel=T(1.0e-12), ftol_abs=T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    return LinearMixedModel(
        form,
        reterms,
        Xy,
        feterm,
        sqrtwts,
        mkparmap(reterms),
        (n=length(y), p=feterm.rank, nretrms=length(reterms)),
        A,
        L,
        optsum,
        isnothing(σ) ? nothing : T(σ),
    )
end

function fit(::Type{LinearMixedModel}, f::FormulaTerm, tbl;
             wts=[], contrasts=Dict{Symbol,Any}(), progress::Bool=true,
             REML::Bool=false, σ=nothing)
    return fit(LinearMixedModel, f, Tables.columntable(tbl);
               wts, contrasts, progress, REML, σ)
end

function fit(::Type{LinearMixedModel}, f::FormulaTerm, tbl::Tables.ColumnTable;
             wts=[], contrasts=Dict{Symbol,Any}(), progress=true, REML=false, σ=nothing)
    return fit!(LinearMixedModel(f, tbl; contrasts, wts, σ); progress, REML)
end

function _offseterr()
    return throw(
        ArgumentError(
            "Offsets are not supported in linear models. You can simply shift the response by the offset.",
        ),
    )
end

function fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl;
    wts=[],
    contrasts=Dict{Symbol,Any}(),
    progress::Bool=true,
    REML::Bool=false,
    offset=[],
    σ=nothing,
)
    return if !isempty(offset)
        _offseterr()
    else
        fit(LinearMixedModel, f, tbl; wts, contrasts, progress, REML, σ)
    end
end

function fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl,
    d::Normal,
    l::IdentityLink;
    wts=[],
    contrasts=Dict{Symbol,Any}(),
    progress::Bool=true,
    REML::Bool=false,
    offset=[],
    σ=nothing,
    fast::Bool=false,
    nAGQ::Integer=1,
)
    return if !isempty(offset)
        _offseterr()
    else
        fit(LinearMixedModel, f, tbl; wts, contrasts, progress, REML, σ)
    end
end

function StatsBase.coef(m::LinearMixedModel{T}) where {T}
    piv = m.feterm.piv
    return invpermute!(fixef!(similar(piv, T), m), piv)
end

βs(m::LinearMixedModel) = NamedTuple{(Symbol.(coefnames(m))...,)}(coef(m))

function StatsBase.coefnames(m::LinearMixedModel)
    Xtrm = m.feterm
    return invpermute!(copy(Xtrm.cnames), Xtrm.piv)
end

function StatsBase.coeftable(m::LinearMixedModel)
    co = coef(m)
    se = stderror!(similar(co), m)
    z = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    names = coefnames(m)

    return CoefTable(
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
        spL = LowerTriangular(SparseMatrixCSC{T,Int}(sparseL(m)))
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
    return val
end

function _cvtbl(arr::Array{T,3}, trm) where {T}
    return merge(
        NamedTuple{(fname(trm),)}((trm.levels,)),
        columntable([
            NamedTuple{(:σ, :ρ)}(sdcorr(view(arr, :, :, i))) for i in axes(arr, 3)
        ]),
    )
end

"""
    condVartables(m::LinearMixedModel)

Return the conditional covariance matrices of the random effects as a `NamedTuple` of columntables
"""
function condVartables(m::MixedModel{T}) where {T}
    return NamedTuple{fnames(m)}((map(_cvtbl, condVar(m), m.reterms)...,))
end

function pushALblock!(A, L, blk)
    push!(L, blk)
    return push!(A, deepcopy(isa(blk, BlockedSparse) ? blk.cscmat : blk))
end

function createAL(reterms::Vector{AbstractReMat{T}}, Xy::FeMat{T}) where {T}
    k = length(reterms)
    vlen = kchoose2(k + 1)
    A = sizehint!(AbstractMatrix{T}[], vlen)
    L = sizehint!(AbstractMatrix{T}[], vlen)
    for i in eachindex(reterms)
        for j in 1:i
            pushALblock!(A, L, densify(reterms[i]' * reterms[j]))
        end
    end
    for j in eachindex(reterms)   # can't fold this into the previous loop b/c block order
        pushALblock!(A, L, densify(Xy' * reterms[j]))
    end
    pushALblock!(A, L, densify(Xy'Xy))
    for i in 2:k      # check for fill-in due to non-nested grouping factors
        ci = reterms[i]
        for j in 1:(i - 1)
            cj = reterms[j]
            if !isnested(cj, ci)
                for l in i:k
                    ind = block(l, i)
                    L[ind] = Matrix(L[ind])
                end
                break
            end
        end
    end
    return A, L
end

StatsBase.deviance(m::LinearMixedModel) = objective(m)

GLM.dispersion(m::LinearMixedModel, sqr::Bool=false) = sqr ? varest(m) : sdest(m)

GLM.dispersion_parameter(m::LinearMixedModel) = true

"""
    feL(m::LinearMixedModel)

Return the lower Cholesky factor for the fixed-effects parameters, as an `LowerTriangular`
`p × p` matrix.
"""
function feL(m::LinearMixedModel)
    XyL = m.L[end]
    k = size(XyL, 1)
    inds = Base.OneTo(k - 1)
    return LowerTriangular(view(XyL, inds, inds))
end

"""
    fit!(m::LinearMixedModel[; progress::Bool=true, REML::Bool=false])

Optimize the objective of a `LinearMixedModel`.  When `progress` is `true` a
`ProgressMeter.ProgressUnknown` display is shown during the optimization of the
objective, if the optimization takes more than one second or so.
"""
function fit!(m::LinearMixedModel{T}; progress::Bool=true, REML::Bool=false) where {T}
    optsum = m.optsum
    # this doesn't matter for LMM, but it does for GLMM, so let's be consistent
    if optsum.feval > 0
        throw(ArgumentError("This model has already been fitted. Use refit!() instead."))
    end
    opt = Opt(optsum)
    optsum.REML = REML
    prog = ProgressUnknown("Minimizing"; showspeed=true)
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = objective(updateL!(setθ!(m, x)))
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        return val
    end
    NLopt.min_objective!(opt, obj)
    optsum.finitial = obj(optsum.initial, T[])
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)
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
    _check_nlopt_return(ret)
    return m
end

function fitted!(v::AbstractArray{T}, m::LinearMixedModel{T}) where {T}
    ## FIXME: Create and use `effects(m) -> β, b` w/o calculating β twice
    Xtrm = m.feterm
    vv = mul!(vec(v), Xtrm.x, fixef!(similar(Xtrm.piv, T), m))
    for (rt, bb) in zip(m.reterms, ranef(m))
        mul!(vv, rt, bb, one(T), one(T))
    end
    return v
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
    Xtrm = m.feterm
    fill!(v, -zero(T))
    XyL = m.L[end]
    L = feL(m)
    k = size(XyL, 1)
    r = size(L, 1)
    for j in 1:r
        v[j] = XyL[k, j]
    end
    ldiv!(feL(m)', length(v) == r ? v : view(v, 1:r))
    return v
end

"""
    fixef(m::MixedModel)

Return the fixed-effects parameter vector estimate of `m`.

In the rank-deficient case the truncated parameter vector, of length `rank(m)` is returned.
This is unlike `coef` which always returns a vector whose length matches the number of
columns in `X`.
"""
fixef(m::LinearMixedModel{T}) where {T} = fixef!(Vector{T}(undef, m.feterm.rank), m)

"""
    fixefnames(m::MixedModel)

Return a (permuted and truncated in the rank-deficient case) vector of coefficient names.
"""
function fixefnames(m::LinearMixedModel{T}) where {T}
    Xtrm = m.feterm
    return Xtrm.cnames[1:(Xtrm.rank)]
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
        throw(
            DimensionMismatch(
                "length(v) = $(length(v)) ≠ length(m.parmap) = $(length(pmap))"
            ),
        )
    end
    reind = 1
    λ = first(m.reterms).λ
    for (k, tp) in enumerate(pmap)
        tp1 = first(tp)
        if reind ≠ tp1
            reind = tp1
            λ = m.reterms[tp1].λ
        end
        v[k] = λ[tp[2], tp[3]]
    end
    return v
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
        vcov(m; corr=true)
    elseif s == :vcov
        vcov(m; corr=false)
    elseif s == :PCA
        NamedTuple{fnames(m)}(PCA.(m.reterms))
    elseif s == :pvalues
        ccdf.(Chisq(1), abs2.(coef(m) ./ stderror(m)))
    elseif s == :stderror
        stderror(m)
    elseif s == :u
        ranef(m; uscale=true)
    elseif s == :lowerbd
        m.optsum.lowerbd
    elseif s == :X
        modelmatrix(m)
    elseif s == :y
        let xy = m.Xymat.xy
            view(xy, :, size(xy, 2))
        end
    elseif s == :rePCA
        rePCA(m)
    else
        getfield(m, s)
    end
end

StatsBase.islinear(m::LinearMixedModel) = true

function StatsBase.leverage(m::LinearMixedModel{T}) where {T}
    # This can be done more efficiently but reusing existing tools is easier.
    # The i'th leverage value is obtained by replacing the response with the i'th
    # basis vector, updating A and L, then taking the sum of squared values of the
    # last row of L, excluding the last position.
    yview = m.y
    yorig = copy(yview)
    kp1 = length(m.reterms) + 1
    L = m.L
    lastL = last(L)
    pp1 = size(lastL, 1)
    p = pp1 - 1
    value = map(eachindex(yorig)) do i
        fill!(yview, zero(T))
        yview[i] = one(T)
        updateL!(reevaluateAend!(m))
        s = sum(abs2, view(lastL, pp1, Base.OneTo(p)))
        for j in eachindex(m.reterms)
            Lblock = m.L[block(kp1, j)]
            s += sum(abs2, view(Lblock, pp1, :))
        end
        s
    end
    copyto!(m.y, yorig)
    updateL!(reevaluateAend!(m))
    return value
end

function StatsBase.loglikelihood(m::LinearMixedModel)
    if m.optsum.REML
        throw(ArgumentError("loglikelihood not available for models fit by REML"))
    end
    return -objective(m) / 2
end

lowerbd(m::LinearMixedModel) = m.optsum.lowerbd

function mkparmap(reterms::Vector{AbstractReMat{T}}) where {T}
    parmap = NTuple{3,Int}[]
    for (k, trm) in enumerate(reterms)
        n = LinearAlgebra.checksquare(trm.λ)
        for ind in trm.inds
            d, r = divrem(ind - 1, n)
            push!(parmap, (k, r + 1, d + 1))
        end
    end
    return parmap
end

nθ(m::LinearMixedModel) = length(m.parmap)

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m`
"""
function objective(m::LinearMixedModel{T}) where {T}
    wts = m.sqrtwts
    denomdf = T(ssqdenom(m))
    σ = getfield(m, :σ)
    val = if σ === nothing
        logdet(m) + denomdf * (one(T) + log2π + log(pwrss(m) / denomdf))
    else
        denomdf * (log2π + 2 * log(σ)) + logdet(m) + pwrss(m) / σ^2
    end
    return isempty(wts) ? val : val - T(2.0) * sum(log, wts)
end

function Base.propertynames(m::LinearMixedModel, private::Bool=false)
    return (
        fieldnames(LinearMixedModel)...,
        :θ,
        :theta,
        :β,
        :beta,
        :βs,
        :betas,
        :λ,
        :lambda,
        :stderror,
        :σ,
        :sigma,
        :σs,
        :sigmas,
        :σρs,
        :sigmarhos,
        :b,
        :u,
        :lowerbd,
        :X,
        :y,
        :corr,
        :vcov,
        :PCA,
        :rePCA,
        :objective,
        :pvalues,
    )
end

"""
    pwrss(m::LinearMixedModel)

The penalized, weighted residual sum-of-squares.
"""
pwrss(m::LinearMixedModel) = abs2(last(last(m.L)))

"""
    ranef!(v::Vector{Matrix{T}}, m::MixedModel{T}, β, uscale::Bool) where {T}

Overwrite `v` with the conditional modes of the random effects for `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise
on the original scale
"""
function ranef!(
    v::Vector, m::LinearMixedModel{T}, β::AbstractArray{T}, uscale::Bool
) where {T}
    (k = length(v)) == length(m.reterms) || throw(DimensionMismatch(""))
    L = m.L
    lind = length(L)
    for j in k:-1:1
        lind -= 1
        Ljkp1 = L[lind]
        vj = v[j]
        length(vj) == size(Ljkp1, 2) || throw(DimensionMismatch(""))
        pp1 = size(Ljkp1, 1)
        copyto!(vj, view(Ljkp1, pp1, :))
        mul!(vec(vj), view(Ljkp1, 1:(pp1 - 1), :)', β, -one(T), one(T))
    end
    for i in k:-1:1
        Lii = L[kp1choose2(i)]
        vi = vec(v[i])
        ldiv!(adjoint(isa(Lii, Diagonal) ? Lii : LowerTriangular(Lii)), vi)
        for j in 1:(i - 1)
            mul!(vec(v[j]), L[block(i, j)]', vi, -one(T), one(T))
        end
    end
    if !uscale
        for (t, vv) in zip(m.reterms, v)
            lmul!(t.λ, vv)
        end
    end
    return v
end

ranef!(v::Vector, m::LinearMixedModel, uscale::Bool) = ranef!(v, m, fixef(m), uscale)

"""
    ranef(m::LinearMixedModel; uscale=false)

Return, as a `Vector{Matrix{T}}`, the conditional modes of the random effects in model `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on
the original scale.

For a named variant, see [`raneftables`](@ref).
"""
function ranef(m::LinearMixedModel{T}; uscale=false) where {T}
    reterms = m.reterms
    v = [Matrix{T}(undef, size(t.z, 1), nlevs(t)) for t in reterms]
    return ranef!(v, m, uscale)
end

LinearAlgebra.rank(m::LinearMixedModel) = m.feterm.rank

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
    return NamedTuple{fnames(m)}(getproperty.(pca, :cumvar))
end

"""
    PCA(m::LinearMixedModel; corr::Bool=true)

Return a named tuple of the principal components analysis of the random effects
covariance matrices or correlation matrices when `corr` is `true`.
"""

function PCA(m::LinearMixedModel; corr::Bool=true)
    return NamedTuple{fnames(m)}(PCA.(m.reterms, corr=corr))
end

"""
    reevaluateAend!(m::LinearMixedModel)

Reevaluate the last column of `m.A` from `m.Xymat`.  This function should be called
after updating the response.
"""
function reevaluateAend!(m::LinearMixedModel)
    A = m.A
    reterms = m.reterms
    nre = length(reterms)
    trmn = reweight!(m.Xymat, m.sqrtwts)
    ind = kp1choose2(nre)
    for trm in m.reterms
        ind += 1
        mul!(A[ind], trmn', trm)
    end
    mul!(A[end], trmn', trmn)
    return m
end

"""
    refit!(m::LinearMixedModel[, y::Vector]; REML=m.optsum.REML, kwargs...)

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.
`kwargs` are the same as [`fit!`](@ref).
"""
function refit!(m::LinearMixedModel; REML=m.optsum.REML, kwargs...)
    return fit!(unfit!(m); REML=REML, kwargs...)
end

function refit!(m::LinearMixedModel, y; kwargs...)
    resp = m.y
    length(y) == length(resp) || throw(DimensionMismatch(""))
    copyto!(resp, y)
    return refit!(m; kwargs...)
end

"""
    restoreoptsum!(m::LinearMixedModel, io::IO)
    restoreoptsum!(m::LinearMixedModel, fnm::AbstractString)

Read, check, and restore the `optsum` field from a JSON stream or filename.
"""
function restoreoptsum!(m::LinearMixedModel, io::IO)
    dict = JSON3.read(io)
    ops = m.optsum
    okay =
        (setdiff(propertynames(ops), keys(dict)) == [:lowerbd]) &&
        all(ops.lowerbd .≤ dict.initial) &&
        all(ops.lowerbd .≤ dict.final)
    if !okay
        throw(ArgumentError("initial or final parameters in io do not satify lowerbd"))
    end
    for fld in (:feval, :finitial, :fmin, :ftol_rel, :ftol_abs, :maxfeval, :nAGQ, :REML)
        setproperty!(ops, fld, getproperty(dict, fld))
    end
    ops.initial_step = copy(dict.initial_step)
    ops.xtol_rel = copy(dict.xtol_rel)
    copyto!(ops.initial, dict.initial)
    copyto!(ops.final, dict.final)
    for (v, f) in (:initial => :finitial, :final => :fmin)
        if !isapprox(objective(updateL!(setθ!(m, getfield(ops, v)))), getfield(ops, f))
            throw(ArgumentError("model m at $v does not give stored $f"))
        end
    end
    ops.optimizer = Symbol(dict.optimizer)
    ops.returnvalue = Symbol(dict.returnvalue)
    return m
end

function restoreoptsum!(m::LinearMixedModel, fnm::AbstractString)
    open(fnm, "r") do io
        restoreoptsum!(m, io)
    end
end

function reweight!(m::LinearMixedModel, weights)
    sqrtwts = map!(sqrt, m.sqrtwts, weights)
    reweight!.(m.reterms, Ref(sqrtwts))
    reweight!(m.Xymat, sqrtwts)
    updateA!(m)
    return updateL!(m)
end

"""
    saveoptsum(io::IO, m::LinearMixedModel)
    saveoptsum(fnm::AbstractString, m::LinearMixedModel)

Save `m.optsum` (w/o the `lowerbd` field) in JSON format to an IO stream or a file

The reason for omitting the `lowerbd` field is because it often contains `-Inf`
values that are not allowed in JSON.
"""
saveoptsum(io::IO, m::LinearMixedModel) = JSON3.write(io, m.optsum)
function saveoptsum(fnm::AbstractString, m::LinearMixedModel)
    open(fnm, "w") do io
        saveoptsum(io, m)
    end
end

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest(m::LinearMixedModel) = something(getfield(m, :σ),  √varest(m))

"""
    setθ!(m::LinearMixedModel, v)

Install `v` as the θ parameters in `m`.
"""
function setθ!(m::LinearMixedModel{T}, θ::AbstractVector) where {T}
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
    return m
end

function Base.setproperty!(m::LinearMixedModel, s::Symbol, y)
    return s == :θ ? setθ!(m, y) : setfield!(m, s, y)
end

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
    return show(io, coeftable(m))
end

Base.show(io::IO, m::LinearMixedModel) = Base.show(io, MIME"text/plain"(), m)

"""
    _coord(A::AbstractMatrix)

Return the positions and values of the nonzeros in `A` as a
`NamedTuple{(:i, :j, :v), Tuple{Vector{Int32}, Vector{Int32}, Vector{Float64}}}`
"""
function _coord(A::Diagonal)
    return (i=Int32.(axes(A, 1)), j=Int32.(axes(A, 2)), v=A.diag)
end

function _coord(A::UniformBlockDiagonal)
    dat = A.data
    r, c, k = size(dat)
    blk = repeat(r .* (0:(k - 1)); inner=r * c)
    return (
        i=Int32.(repeat(1:r; outer=c * k) .+ blk),
        j=Int32.(repeat(1:c; inner=r, outer=k) .+ blk),
        v=vec(dat),
    )
end

function _coord(A::SparseMatrixCSC{T,Int32}) where {T}
    rv = rowvals(A)
    cv = similar(rv)
    for j in axes(A, 2), k in nzrange(A, j)
        cv[k] = j
    end
    return (i=rv, j=cv, v=nonzeros(A))
end

_coord(A::BlockedSparse) = _coord(A.cscmat)

function _coord(A::Matrix)
    m, n = size(A)
    return (
        i=Int32.(repeat(axes(A, 1); outer=n)),
        j=Int32.(repeat(axes(A, 2); inner=m)),
        v=vec(A),
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
    val = (i=Int32[], j=Int32[], v=T[])
    for i in 1:nt, j in 1:i
        Lblk = L[block(i, j)]
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
    return dropzeros!(tril!(sparse(val...)))
end

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
    return m.optsum.REML ? n - m.dims.p : n
end

"""
    std(m::MixedModel)

Return the estimated standard deviations of the random effects as a `Vector{Vector{T}}`.

FIXME: This uses an old convention of isfinite(sdest(m)).  Probably drop in favor of m.σs
"""
function Statistics.std(m::LinearMixedModel)
    rl = rowlengths.(m.reterms)
    s = sdest(m)
    return isfinite(s) ? rmul!(push!(rl, [1.0]), s) : rl
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
    invpermute!(v, m.feterm.piv)
    return v
end

function StatsBase.stderror(m::LinearMixedModel{T}) where {T}
    return stderror!(similar(m.feterm.piv, T), m)
end

"""
    updateA!(m::LinearMixedModel)

Update the cross-product array, `m.A`, from `m.reterms` and `m.Xymat`

This is usually done after a reweight! operation.
"""
function updateA!(m::LinearMixedModel)
    reterms = m.reterms
    k = length(reterms)
    A = m.A
    ind = 1
    for (i, trmi) in enumerate(reterms)
        for j in 1:i
            mul!(A[ind], trmi', reterms[j])
            ind += 1
        end
    end
    Xymattr = adjoint(m.Xymat)
    for trm in reterms
        mul!(A[ind], Xymattr, trm)
        ind += 1
    end
    mul!(A[end], Xymattr, m.Xymat)
    return m
end

"""
    unfit!(model::MixedModel)

Mark a model as unfitted.
"""
function unfit!(model::LinearMixedModel{T}) where {T}
    model.optsum.feval = -1
    model.optsum.initial_step = T[]
    reevaluateAend!(model)

    return model
end

"""
    updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.reterms` (used for λ only)

This is the crucial step in evaluating the objective, given a new parameter value.
"""
function updateL!(m::LinearMixedModel{T}) where {T}
    A, L, reterms = m.A, m.L, m.reterms
    k = length(reterms)
    for (l, a) in zip(L, A) # copy each element of A to corresponding element of L
        copyto!(l, a)       # For some reason copyto!.(L, A) allocates a lot of memory
    end
    for j in eachindex(reterms) # pre- and post-multiply by Λ, add I to diagonal
        cj = reterms[j]
        scaleinflate!(L[kp1choose2(j)], cj)
        for i in (j + 1):(k + 1)     # postmultiply column by Λ
            rmulΛ!(L[block(i, j)], cj)
        end
        for jj in 1:(j - 1)        # premultiply row by Λ'
            lmulΛ!(cj', L[block(j, jj)])
        end
    end
    for j in 1:(k + 1)             # blocked Cholesky
        Ljj = L[kp1choose2(j)]
        for jj in 1:(j - 1)
            rankUpdate!(Hermitian(Ljj, :L), L[block(j, jj)], -one(T), one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i in (j + 1):(k + 1)
            Lij = L[block(i, j)]
            for jj in 1:(j - 1)
                mul!(Lij, L[block(i, jj)], L[block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LjjT')
        end
    end
    return m
end

"""
    varest(m::LinearMixedModel)

Returns the estimate of σ², the variance of the conditional distribution of Y given B.
"""
varest(m::LinearMixedModel) = isnothing(getfield(m, :σ)) ?  pwrss(m) / ssqdenom(m) : getfield(m, :σ)

function StatsBase.weights(m::LinearMixedModel)
    rtwts = m.sqrtwts
    return isempty(rtwts) ? ones(eltype(rtwts), nobs(m)) : abs2.(rtwts)
end
