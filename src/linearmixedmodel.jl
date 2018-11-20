"""
    LinearMixedModel

Linear mixed-effects model representation

## Fields

* `formula`: the formula for the model
* `trms`: a `Vector` of `AbstractTerm` types representing the model.  The last element is the response.
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
    lbd = T[]
    θ = T[]
    for c in filter(x -> isa(x, ReMat), cols)
        append!(lbd, lowerbd(c))
        append!(θ, getθ(c))
    end
    optsum = OptSummary(θ, lbd, :LN_BOBYQA;
        ftol_rel = T(1.0e-12), ftol_abs = T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(form, cols, T[], A, L, optsum)
end

function Base.getproperty(m::LinearMixedModel, s::Symbol)
    if s ∈ (:θ, :theta)
        getθ(m)
    elseif s ∈ (:β, :beta)
        fixef(m)
    elseif s ∈ (:λ, :lambda)
        getλ(m)
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

Base.setproperty!(m::LinearMixedModel, s::Symbol, y) = s == :θ ? setθ!(m, y) : setfield!(m, s, y)

Base.propertynames(m::LinearMixedModel, private=false) =
    (:formula, :trms, :A, :L, :optsum, :θ, :theta, :β, :beta, :λ, :lambda, :σ, :sigma, :b, :u, :lowerbd, :X, :y, :rePCA)

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
        if isa(cols[j], ReMat)   # for ReMat terms
            λ = cols[j].λ
            for i in j:k         # postmultiply column by Λ
                rmulΛ!(L[Block(i, j)], λ)
            end
            for jj in 1:j        # premultiply row by Λ'
                lmulλ!(λ', L[Block(j, jj)])
            end
            Ljj += I  # inflate the diagonal of the diagonal block(check if this allocates)
        end
        for jj in 1:(j - 1)
            rankUpdate!(Hermitian(Ljj, :L), L[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        for i in (j + 1):k
            for jj in 1:(j - 1)
                mulαβ!(Lij, L[Block(i, jj)], L[Block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LowerTriangular(Ljj)')
        end
    end
    m
end

