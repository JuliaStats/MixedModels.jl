"""
    LinearMixedModel

Linear mixed-effects model representation

## Fields

* `formula`: the formula for the model
* `reterms`: a `Vector{ReMat{T}}` of random-effects terms.
* `feterms`: a `Vector{FeMat{T}}` of the fixed-effects model matrix and the response
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
* `u`: random effects on the orthogonal scale, as a vector of matrices
* `lowerbd`: lower bounds on the elements of θ
* `X`: the fixed-effects model matrix
* `y`: the response vector
"""
struct LinearMixedModel{T <: AbstractFloat} <: MixedModel{T}
    formula::FormulaTerm
    reterms::Vector{ReMat{T}}
    feterms::Vector{FeMat{T}}
    sqrtwts::Vector{T}
    A::BlockMatrix{T}            # cross-product blocks
    L::BlockMatrix{T}
    optsum::OptSummary{T}
end
LinearMixedModel(f::FormulaTerm, tbl;
                 contrasts = Dict{Symbol,Any}(),
                 wts = []) =
    LinearMixedModel(f::FormulaTerm, Tables.columntable(tbl),
                     contrasts = contrasts,
                     wts = wts)
function LinearMixedModel(f::FormulaTerm, tbl::Tables.ColumnTable;
                          contrasts = Dict{Symbol,Any}(),
                          wts = [])
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
    for (i,x) in enumerate(Xs)
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

    sort!(reterms, by=nranef, rev=true)

    # create A and L
    terms = vcat(reterms, feterms)
    k = length(terms)
    sz = append!(size.(reterms, 2), rank.(feterms))
    A = BlockArray(undef_blocks, AbstractMatrix{T}, sz, sz)
    L = BlockArray(undef_blocks, AbstractMatrix{T}, sz, sz)
    for j in 1:k
        for i in j:k
            Lij = L[Block(i,j)] = densify(terms[i]'terms[j])
            A[Block(i,j)] = deepcopy(isa(Lij, BlockedSparse) ? Lij.cscmat : Lij)
        end
    end
    for i in 2:length(reterms) # check for fill-in due to non-nested grouping factors
        ci = reterms[i]
        for j in 1:(i - 1)
            cj = reterms[j]
            if !isnested(cj, ci)
                for l in i:k
                    L[Block(l, i)] = Matrix(L[Block(l, i)])
                end
                break
            end
        end
    end
    lbd = foldl(vcat, lowerbd(c) for c in reterms)
    θ = foldl(vcat, getθ(c) for c in reterms)
    optsum = OptSummary(θ, lbd, :LN_BOBYQA, ftol_rel = T(1.0e-12), ftol_abs = T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(form, reterms, feterms, sqrt.(convert(Vector{T}, wts)), A, L, optsum)
end
fit(::Type{LinearMixedModel}, f::FormulaTerm, tbl;
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    verbose::Bool = false,
    REML::Bool = false) =
    fit(LinearMixedModel, f, Tables.columntable(tbl),
        wts = wts, contrasts = contrasts, verbose = verbose, REML = REML)
fit(::Type{LinearMixedModel},
    f::FormulaTerm,
    tbl::Tables.ColumnTable;
    wts = wts, contrasts = contrasts, verbose = verbose, REML = REML) =
    fit!(LinearMixedModel(f, tbl,
                          contrasts = contrasts,
                          wts = wts),
                          verbose = verbose,
                          REML = REML)

StatsBase.coef(m::MixedModel) = fixef(m, false)

function βs(m::LinearMixedModel)
    fetrm = first(m.feterms)
    (; (k => v for (k,v) in zip(Symbol.(fetrm.cnames), fixef(m)))...)
end

StatsBase.coefnames(m::LinearMixedModel) = first(m.feterms).cnames

function StatsBase.coeftable(m::MixedModel)
    co = coef(m)
    se = stderror(m)
    z = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    CoefTable(hcat(co, se, z, pvalue), ["Estimate", "Std.Error", "z value", "P(>|z|)"],
        first(m.feterms).cnames, 4)
end

"""
    cond(m::MixedModel)

Return a vector of condition numbers of the λ matrices for the random-effects terms
"""
LinearAlgebra.cond(m::MixedModel) = cond.(m.λ)

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
    if !isone(length(retrms)) || !isa(L11, Diagonal{T, Vector{T}})
        throw(ArgumentError("code for vector-valued r.e. or more than one term not yet written"))
    end
    ll = first(t1.λ)
    Ld = L11.diag
    Array{T, 3}[reshape(abs2.(ll ./ Ld) .* varest(m), (1, 1, length(Ld)))]
end

"""
    describeblocks(io::IO, m::MixedModel)
    describeblocks(m::MixedModel)

Describe the types and sizes of the blocks in the lower triangle of `m.A` and `m.L`.
"""
function describeblocks(io::IO, m::LinearMixedModel)
    A = m.A
    L = m.L
    for i in 1:BlockArrays.nblocks(A, 2), j in 1:i
        println(io, i, ",", j, ": ", typeof(A[Block(i, j)]), " ",
            BlockArrays.blocksize(A, (i, j)), " ", typeof(L[Block(i, j)]))
    end
end
describeblocks(m::MixedModel) = describeblocks(stdout, m)

StatsBase.deviance(m::MixedModel) = objective(m)

GLM.dispersion(m::LinearMixedModel, sqr::Bool=false) = sqr ? varest(m) : sdest(m)

GLM.dispersion_parameter(m::LinearMixedModel) = true

StatsBase.dof(m::LinearMixedModel) = size(m)[2] + sum(nθ, m.reterms) + 1

function StatsBase.dof_residual(m::LinearMixedModel)::Int
    (n, p, q, k) = size(m)
    n - m.optsum.REML * p
end

"""
    feL(m::MixedModel)

Return the lower Cholesky factor for the fixed-effects parameters, as an `LowerTriangular`
`p × p` matrix.
"""
feL(m::LinearMixedModel) = LowerTriangular(m.L.blocks[end - 1, end - 1])

"""
    fit!(m::LinearMixedModel[; verbose::Bool=false, REML::Bool=false])

Optimize the objective of a `LinearMixedModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
"""
function fit!(m::LinearMixedModel{T}; verbose::Bool=false, REML::Bool=false) where {T}
    optsum = m.optsum
    opt = Opt(optsum)
    feval = 0
    optsum.REML = REML
    function obj(x, g)
        isempty(g) || error("gradient not defined")
        feval += 1
        val = objective(updateL!(setθ!(m, x)))
        feval == 1 && (optsum.finitial = val)
        verbose && println("f_", feval, ": ", round(val, digits=5), " ", x)
        val
    end
    NLopt.min_objective!(opt, obj)
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

    optsum.feval = feval
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    ret == :ROUNDOFF_LIMITED && @warn("NLopt was roundoff limited")
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXFEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end

function fitted!(v::AbstractArray{T}, m::LinearMixedModel{T}) where {T}
    ## FIXME: Create and use `effects(m) -> β, b` w/o calculating β twice
    vv = vec(v)
    mul!(vv, first(m.feterms), fixef(m))
    for (rt, bb) in zip(m.reterms, ranef(m))
        unscaledre!(vv, rt, bb)
    end
    v
end

StatsBase.fitted(m::LinearMixedModel{T}) where {T} = fitted!(Vector{T}(undef, nobs(m)), m)

"""
    fixef!(v::Vector{T}, m::LinearMixedModel{T})

Overwrite `v` with the pivoted and, possibly, truncated fixed-effects coefficients of model `m`
"""
fixef!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T} =
    ldiv!(feL(m)', copyto!(v, m.L.blocks[end, end - 1]))

"""
    fixef(m::MixedModel, permuted=true)

Return the fixed-effects parameter vector estimate of `m`.

If `permuted` is `true` the vector elements are permuted according to
`m.trms[end - 1].piv` and truncated to the rank of that term.
"""
function fixef(m::LinearMixedModel{T}, permuted=true) where {T}
    val = ldiv!(feL(m)', vec(copy(m.L.blocks[end, end-1])))
    if !permuted
        Xtrm = first(m.feterms)
        piv = Xtrm.piv
        p = length(piv)
        if Xtrm.rank < p
            val = copyto!(fill(-zero(T), p), val)
        end
        invpermute!(val, piv)
    end
    val
end

"""
    fnames(m::MixedModel)

Return the names of the grouping factors for the random-effects terms.
"""
fnames(m::MixedModel) = ((Symbol(tr.trm.sym) for tr in m.reterms)...,)

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

function Base.getproperty(m::LinearMixedModel, s::Symbol)
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

function StatsBase.loglikelihood(m::LinearMixedModel)
    if m.optsum.REML
        throw(ArgumentError("loglikelihood not available for models fit by REML"))
    end
    -objective(m) / 2
end

lowerbd(m::LinearMixedModel) = m.optsum.lowerbd

"""
    likelihoodratiotest(m::LinearMixedModel...)

Likeihood ratio test applied to a set of nested models.

Note that nesting of the models is not checked.  It is incumbent on the user to check this.
"""
function likelihoodratiotest(m::LinearMixedModel...)
    m = collect(m)   # change the tuple to an array
    dofs = dof.(m)
    ord = sortperm(dofs)
    dofs = dofs[ord]
    devs = deviance.(m)[ord]
    dofdiffs = diff(dofs)
    devdiffs = .-(diff(devs))
    pvals = ccdf.(Chisq.(dofdiffs), devdiffs)
    (models=(dof=dofs, deviance=devs), tests=(dofdiff=dofdiffs, deviancediff=devdiffs, p_values=pvals))
end

function StatsBase.modelmatrix(m::LinearMixedModel)
    fetrm = first(m.feterms)
    if fetrm.rank == size(fetrm, 2)
        fetrm.x
    else
        fetrm.x[:, invperm(fetrm.piv)]
    end
end

nθ(m::LinearMixedModel) = sum(nθ, m.reterms)

StatsBase.nobs(m::LinearMixedModel) = first(size(m))

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m`
"""
function objective(m::LinearMixedModel)
    wts = m.sqrtwts
    logdet(m) + dof_residual(m)*(1 + log2π + log(varest(m))) - (isempty(wts) ? 0 : 2sum(log, wts))
end

StatsBase.predict(m::LinearMixedModel) = fitted(m)

Base.propertynames(m::LinearMixedModel, private=false) =
    (:formula, :sqrtwts, :A, :L, :optsum, :θ, :theta, :β, :beta, :λ, :lambda, :stderror,
     :σ, :sigma, :σs, :sigmas, :b, :u, :lowerbd, :X, :y, :rePCA, :reterms, :feterms,
     :objective, :pvalues)

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
function ranef!(v::Vector, m::LinearMixedModel{T}, β::AbstractArray{T}, uscale::Bool) where {T}
    (k = length(v)) == length(m.reterms) || throw(DimensionMismatch(""))
    L = m.L
    for j in 1:k
        mulαβ!(vec(copyto!(v[j], L[Block(BlockArrays.nblocks(L, 2), j)])),
            L[Block(k + 1, j)]', β, -one(T), one(T))
    end
    for i in k: -1 :1
        Lii = L[Block(i, i)]
        vi = vec(v[i])
        ldiv!(adjoint(isa(Lii, Diagonal) ? Lii : LowerTriangular(Lii)), vi)
        for j in 1:(i - 1)
            mulαβ!(vec(v[j]), L[Block(i, j)]', vi, -one(T), one(T))
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
    ranef(m::LinearMixedModel; uscale=false) #, named=true)

Return, as a `Vector{Vector{T}}` (`Vector{NamedVector{T}}` if `named=true`),
the conditional modes of the random effects in model `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on
the original scale.
"""
function ranef(m::LinearMixedModel{T}; uscale=false, named=false) where {T}
    v = [Matrix{T}(undef, size(t.z, 1), nlevs(t)) for t in m.reterms]
    ranef!(v, m, uscale)
    named || return v
    vnmd = map(NamedArray, v)
    for (trm, vnm) in zip(m.reterms, vnmd)
        setnames!(vnm, trm.cnames, 1)
        setnames!(vnm, string.(trm.trm.contrasts.levels), 2)
    end
    vnmd
end

LinearAlgebra.rank(m::LinearMixedModel) = first(m.feterms).rank

function rePCA(m::LinearMixedModel{T}) where {T}
    re = m.reterms
    nt = length(re)
    tup = ntuple(i -> normalized_variance_cumsum(re[i].λ), nt)
    NamedTuple{ntuple(i -> re[i].trm.sym, nt), typeof(tup)}(tup)
end

"""
    reevaluateAend!(m::LinearMixedModel)

Reevaluate the last column of `m.A` from `m.feterms`.  This function should be called
after updating the response, `m.feterms[end]`.
"""
function reevaluateAend!(m::LinearMixedModel)
    A = m.A
    ftrms = m.feterms
    trmn = reweight!(last(ftrms), m.sqrtwts)
    nblk = BlockArrays.nblocks(A, 1)
    for (j, trm) in enumerate(vcat(m.reterms, ftrms))
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
    reweight!.(m.feterms, Ref(sqrtwts))
    reweight!.(m.reterms, Ref(sqrtwts))
    updateA!(m)
    updateL!(m)
end

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest(m::LinearMixedModel) = √varest(m)

"""
    setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})

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

Base.setproperty!(m::LinearMixedModel, s::Symbol, y) = s == :θ ? setθ!(m, y) : setfield!(m, s, y)

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
        nums = showoff([-oo/ 2, oo, aic(m), bic(m)])
        fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
        for label in [" logLik", "-2 logLik", "AIC", "BIC"]
            print(io, rpad(lpad(label, (fieldwd + textwidth(label)) >> 1), fieldwd))
        end
        println(io)
        print.(Ref(io), lpad.(nums, fieldwd))
        println(io)
    end
    println(io)

    show(io,VarCorr(m))

    print(io," Number of obs: $n; levels of grouping factors: ")
    join(io, nlevs.(m.reterms), ", ")
    println(io)
    println(io,"\n  Fixed-effects parameters:")
    show(io,coeftable(m))
end

function σs(m::LinearMixedModel)
    σ = sdest(m)
    NamedTuple{fnames(m)}(((σs(t, σ) for t in m.reterms)...,))
end

function σρs(m::LinearMixedModel)
    σ = sdest(m)
    NamedTuple{fnames(m)}(((σρs(t, σ) for t in m.reterms)...,))
end

function Base.size(m::LinearMixedModel)
    n, p = size(first(m.feterms))
    n, p, sum(size.(m.reterms, 2)), length(m.reterms)
end

"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This value is the contents of the `1 × 1` bottom right block of `m.L`
"""
sqrtpwrss(m::LinearMixedModel) = first(m.L.blocks[end, end])

"""
    std(m::MixedModel)

Return the estimated standard deviations of the random effects as a `Vector{Vector{T}}`.
"""
function Statistics.std(m::LinearMixedModel)
    rl = rowlengths.(m.reterms)
    s = sdest(m)
    isfinite(s) ? rmul!(push!(rl, [1.]), s) : rl
end

"""
    updateA!(m::LinearMixedModel)

Update the cross-product array, `m.A`, using `m.reterms` and `m.feterms`

This is usually done after a reweight! operation.
"""
function updateA!(m::LinearMixedModel)
    terms = vcat(m.reterms, m.feterms)
    k = length(terms)
    A = m.A
    for j in 1:k
        for i in j:k
            mul!(A[Block(i, j)], terms[i]', terms[j])
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
    k = BlockArrays.nblocks(A, 2)
    for j in 1:k                         # copy lower triangle of A to L
        for i in j:BlockArrays.nblocks(A, 1)
            copyto!(L[Block(i, j)], A[Block(i, j)])
        end
    end
    for (j, cj) in enumerate(m.reterms)  # pre- and post-multiply by Λ, add I to diagonal
        scaleinflate!(L[Block(j, j)], cj)
        for i in (j+1):k         # postmultiply column by Λ
            rmulΛ!(L[Block(i, j)], cj)
        end
        for jj in 1:(j-1)        # premultiply row by Λ'
            lmulΛ!(cj', L[Block(j, jj)])
        end
    end
    for j in 1:k                         # blocked Cholesky
        Ljj = L[Block(j, j)]
        LjjH = isa(Ljj, Diagonal) ? Ljj : Hermitian(Ljj, :L)
        for jj in 1:(j - 1)
            rankUpdate!(LjjH, L[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i in (j + 1):k
            Lij = L[Block(i, j)]
            for jj in 1:(j - 1)
                mulαβ!(Lij, L[Block(i, jj)], L[Block(j, jj)]', -one(T), one(T))
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
varest(m::LinearMixedModel) = pwrss(m) / dof_residual(m)

function StatsBase.vcov(m::LinearMixedModel{T}) where {T}
    Xtrm = first(m.feterms)
    iperm = invperm(Xtrm.piv)
    p = length(iperm)
    r = Xtrm.rank
    Linv = inv(feL(m))
    permvcov = varest(m) * (Linv'Linv)
    if p == Xtrm.rank
        permvcov[iperm, iperm]
    else
        covmat = fill(zero(T)/zero(T), (p, p))
        for j in 1:r, i in 1:r
            covmat[i,j] = permvcov[i, j]
        end
        covmat[iperm, iperm]
    end
end

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
    optsum.feval == -1

    m
end

zerocorr!(m::LinearMixedModel) = zerocorr!(m, fnames(m))
