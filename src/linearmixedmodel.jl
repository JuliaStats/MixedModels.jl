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

function LinearMixedModel(f::FormulaTerm, d::NamedTuple)
    form = apply_schema(f, schema(d), LinearMixedModel)
    y, Xs = model_cols(form, d)

    y = reshape(float(y), (:, 1)) # y as a floating-point matrix
    T = eltype(y)

    reterms = ReMat{T}[]
    feterms = FeMat{T}[]
    for x in Xs
        if isa(x, ReMat{T})
            push!(reterms, x)
        else
            push!(feterms, FeMat(x, String[]))
        end
    end
    push!(feterms, FeMat(y, String[]))
    sort!(reterms, by=nranef, rev=true)

    # create A and L
    sz = append!(size.(reterms, 2), size.(feterms, 2))
    A = BlockArrays._BlockArray(AbstractMatrix{T}, sz, sz)
    L = BlockArrays._BlockArray(AbstractMatrix{T}, sz, sz)
    q = length(reterms)
    k = q + length(feterms)
    for j in 1:k
        cj = j ≤ q ? reterms[j] : feterms[j - q]
        for i in j:k
            Lij = L[Block(i,j)] = densify((i ≤ q ? reterms[i] : feterms[i - q])'cj)
            A[Block(i,j)] = deepcopy(isa(Lij, BlockedSparse) ? Lij.cscmat : Lij)
        end
    end
    for i in 2:q            # check for fill-in due to non-nested grouping factors
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
    lbd = reduce(append!,  lowerbd(c) for c in reterms)
    θ = reduce(append!, getθ(c) for c in reterms)
    optsum = OptSummary(θ, lbd, :LN_BOBYQA, ftol_rel = T(1.0e-12), ftol_abs = T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(form, reterms, feterms, T[], A, L, optsum)
end

StatsBase.coef(m::MixedModel) = fixef(m, false)

"""
    cond(m::MixedModel)

Return a vector of condition numbers of the λ matrices for the random-effects terms
"""
LinearAlgebra.cond(m::MixedModel) = [cond(c.λ) for c in m.reterms]


"""
    describeblocks(io::IO, m::MixedModel)
    describeblocks(m::MixedModel)

Describe the types and sizes of the blocks in the lower triangle of `m.A` and `m.L`.
"""
function describeblocks(io::IO, m::LinearMixedModel)
    A = m.A
    L = m.L
    for i in 1:nblocks(A, 2), j in 1:i
        println(io, i, ",", j, ": ", typeof(A[Block(i, j)]), " ",
                blocksize(A, (i, j)), " ", typeof(L[Block(i, j)]))
    end
end
describeblocks(m::MixedModel) = describeblocks(stdout, m)

StatsBase.dof(m::LinearMixedModel) = size(m)[2] + sum(nθ, m.reterms) + 1

"""
    feL(m::MixedModel)

Return the lower Cholesky factor for the fixed-effects parameters, as an `LowerTriangular`
`p × p` matrix.
"""
feL(m::LinearMixedModel) = LowerTriangular(m.L.blocks[end - 1, end - 1])

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
    fixef!(v::Vector{T}, m::LinearMixedModel{T})

Overwrite `v` with the pivoted and, possibly, truncated fixed-effects coefficients of model `m`
"""
function fixef!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    L = feL(m)
    length(v) == size(L, 1) || throw(DimensionMismatch(""))
    ldiv!(adjoint(L), copyto!(v, m.L.blocks[end, end - 1]))
end

"""
    fixef(m::MixedModel, permuted=true)

Return the fixed-effects parameter vector estimate of `m`.

If `permuted` is `true` the vector elements are permuted according to
`m.trms[end - 1].piv` and truncated to the rank of that term.
"""
function fixef(m::LinearMixedModel{T}, permuted=true) where {T}
    permuted && return fixef!(Vector{T}(undef, size(m)[2]), m)
    Xtrm = first(m.feterms)
    piv = Xtrm.piv
    v = fill(-zero(T), size(piv))
    fixef!(view(v, 1:Xtrm.rank), m)
    invpermute!(v, piv)
end

"""
    getθ(m::LinearMixedModel)

Return the current covariance parameter vector.
"""
getθ(m::LinearMixedModel{T}) where {T} = reduce(append!, getθ.(m.reterms))

function Base.getproperty(m::LinearMixedModel, s::Symbol)
    if s ∈ (:θ, :theta)
        getθ(m)
    elseif s ∈ (:β, :beta)
        fixef(m)
    elseif s ∈ (:λ, :lambda)
        getfield.(m.reterms, :λ)
    elseif s ∈ (:σ, :sigma)
        sdest(m)
    elseif s == :b
        ranef(m)
    elseif s == :u
        ranef(m, uscale = true)
    elseif s == :lowerbd
        m.optsum.lowerbd
    elseif s == :X
        first(m.feterms).x
    elseif s == :y
        vec(m.feterms[end].x)
    elseif s == :rePCA
        normalized_variance_cumsum.(getλ(m))
    else
        getfield(m, s)
    end
end

StatsBase.loglikelihood(m::LinearMixedModel) = -objective(m) / 2

lowerbd(m::LinearMixedModel) = m.optsum.lowerbd

StatsBase.nobs(m::LinearMixedModel) = first(size(m))

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

StatsBase.response(m::LinearMixedModel) = vec(m.feterms[end].x)

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest(m::LinearMixedModel) = sqrtpwrss(m) / √nobs(m)

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
    updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.reterms` (used for λ only)

This is the crucial step in evaluating the objective, given a new parameter value.
"""
function updateL!(m::LinearMixedModel{T}) where {T}
    A = m.A
    L = m.L
    k = nblocks(A, 2)
    for j in 1:k                         # copy lower triangle of A to L
        for i in j:nblocks(A, 1)
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
varest(m::LinearMixedModel) = pwrss(m) / nobs(m)
