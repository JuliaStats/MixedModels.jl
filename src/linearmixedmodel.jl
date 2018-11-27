"""
    LinearMixedModel

Linear mixed-effects model representation

## Fields

* `formula`: the formula for the model
* `cols`: a `Vector` of `Union{ReMat,FeMat}` representing the model.  The last element is the response.
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
    cols::Vector
    sqrtwts::Vector{T}
    A::BlockMatrix{T}            # cross-product blocks
    L::BlockMatrix{T}
    optsum::OptSummary{T}
end

function LinearMixedModel(f::FormulaTerm, d::NamedTuple)
    form = apply_schema(f, schema(d), LinearMixedModel)
    y, Xs = model_cols(form, d)

    y = reshape(float(y), (:, 1)) # y as a floating-point matrix
    T = eltype(y)

    # might need to add a guard to make sure Xs isn't just a single matrix
    cols = sort!(AbstractMatrix{T}[Xs..., y], by=nranef, rev=true)

    # create A and L
    sz = size.(cols, 2)
    k = length(cols)
    A = BlockArrays._BlockArray(AbstractMatrix{T}, sz, sz)
    L = BlockArrays._BlockArray(AbstractMatrix{T}, sz, sz)
    for j in 1:k
        cj = cols[j]
        for i in j:k
            Lij = L[Block(i,j)] = densify(cols[i]'cj)
            A[Block(i,j)] = deepcopy(isa(Lij, BlockedSparse) ? Lij.cscmat : Lij)
        end
    end
                  # check for fill-in due to non-nested grouping factors
    for i in 2:k
        ci = cols[i]
        if isa(ci, ReMat)
            for j in 1:(i - 1)
                cj = cols[j]
                if isa(cj, ReMat) && !isnested(cj, ci)
                    for k in i:k
                        L[Block(k, i)] = Matrix(L[Block(k, i)])
                    end
                    break
                end
            end
        end
    end
    lbd = reduce(append!,  lowerbd(c) for c in cols if isa(c, ReMat))
    θ = reduce(append!, getθ(c) for c in cols if isa(c, ReMat))
    optsum = OptSummary(θ, lbd, :LN_BOBYQA;
        ftol_rel = T(1.0e-12), ftol_abs = T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(form, cols, T[], A, L, optsum)
end

StatsBase.dof(m::LinearMixedModel) = 
    size(m)[2] + sum(nθ(c) for c in m.cols if isa(c, ReMat)) + 1

"""
    fit!(m::LinearMixedModel[, verbose::Bool=false])

Optimize the objective of a `LinearMixedModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
"""
function StatsBase.fit!(m::LinearMixedModel{T}, verbose::Bool=false) where {T}
    optsum = m.optsum
    opt = Opt(optsum)
    feval = 0
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

"""
    getθ(m::LinearMixedModel)

Return the current covariance parameter vector.
"""
getθ(m::LinearMixedModel{T}) where {T} = 
    reduce(append!, getθ(m) for m in m.cols if isa(m, ReMat))

function Base.getproperty(m::LinearMixedModel, s::Symbol)
    if s ∈ (:θ, :theta)
        getθ(m)
    elseif s ∈ (:β, :beta)
        fixef(m)
    elseif s ∈ (:λ, :lambda)
        [m.λ for m in m.cols if isa(m, ReMat)]
    elseif s ∈ (:σ, :sigma)
        sdest(m)
    elseif s == :b
        ranef(m)
    elseif s == :u
        ranef(m, uscale = true)
    elseif s == :lowerbd
        m.optsum.lowerbd
    elseif s == :X
        m.trms[end - 1].x
    elseif s == :y
        vec(m.trms[end].x)
    elseif s == :rePCA
        normalized_variance_cumsum.(getλ(m))
    else
        getfield(m, s)
    end
end

StatsBase.loglikelihood(m::LinearMixedModel) = -objective(m) / 2

lowerbd(m::LinearMixedModel) = m.optsum.lowerbd

StatsBase.nobs(m::LinearMixedModel) = size(first(m.cols), 1)

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m`
"""
function objective(m::LinearMixedModel)
    wts = m.sqrtwts
    logdet(m) + nobs(m)*(1 + log2π + log(varest(m))) - (isempty(wts) ? 0 : 2sum(log, wts))
end

Base.propertynames(m::LinearMixedModel, private=false) =
    (:formula, :cols, :sqrtwts, :A, :L, :optsum, :θ, :theta, :β, :beta, :λ, :lambda, :σ, :sigma, :b, :u, :lowerbd, :X, :y, :rePCA)

"""
    pwrss(m::LinearMixedModel)

The penalized, weighted residual sum-of-squares.
"""
pwrss(m::LinearMixedModel) = abs2(sqrtpwrss(m))

"""
    setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})

Install `v` as the θ parameters in `m`.
"""
function setθ!(m::LinearMixedModel, v)
    offset = 0
    for trm in m.cols
        if isa(trm, ReMat)
            k = nθ(trm)
            setθ!(trm, view(v, (1:k) .+ offset))
            offset += k
        end
    end
    m
end

Base.setproperty!(m::LinearMixedModel, s::Symbol, y) = s == :θ ? setθ!(m, y) : setfield!(m, s, y)

function Base.size(m::LinearMixedModel)
    cols = m.cols
    n, p = size(cols[end - 1])
    k = sum(isa.(cols, ReMat))
    q = sum(size(c, 2) for c in cols if isa(c, ReMat))
    n, p, q, k
end

"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This value is the contents of the `1 × 1` bottom right block of `m.L`
"""
sqrtpwrss(m::LinearMixedModel) = first(m.L.blocks[end, end])

"""
    updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.cols` (used for λ only)

This is the crucial step in evaluating the objective, given a new parameter value.
"""
function updateL!(m::LinearMixedModel{T}) where {T}
    A = m.A
    L = m.L
    cols = m.cols
    k = length(cols)
    for j in 1:k
        for i in j:k    # copy lower triangle of A to L
            copyto!(L[Block(i, j)], A[Block(i, j)])
        end
        Ljj = L[Block(j, j)]
        cj = cols[j]
        if isa(cj, ReMat)        # for ReMat terms
            scaleinflate!(Ljj, cj)
            for i in (j+1):k         # postmultiply column by Λ
                rmulΛ!(L[Block(i, j)], cj)
            end
            for jj in 1:(j-1)        # premultiply row by Λ'
                lmulΛ!(cj', L[Block(j, jj)])
            end
        end
        for jj in 1:(j - 1)
            rankUpdate!(Hermitian(Ljj, :L), L[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        for i in (j + 1):k
            Lij = L[Block(i, j)]
            for jj in 1:(j - 1)
                mulαβ!(Lij, L[Block(i, jj)], L[Block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LowerTriangular(Ljj)')
        end
    end
    m
end

"""
    varest(m::LinearMixedModel)

Returns the estimate of σ², the variance of the conditional distribution of Y given B.
"""
varest(m::LinearMixedModel) = pwrss(m) / nobs(m)
