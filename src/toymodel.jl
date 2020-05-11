struct ToyModel{T<:AbstractFloat}
    A::Matrix{T}
    L::Matrix{T}
    Λ::AbstractMatrix{T}
    inds::Vector
    Λ̇::Vector
    L̇::Vector{Matrix{T}}
    dims::NamedTuple{(:n, :p, :q, :k, :nθ),NTuple{5,Int64}}
    optsum::OptSummary
end
function ToyModel(m::LinearMixedModel{T}) where {T}
    inds = Λinds(m)
    A = Matrix(Symmetric(copyltri(m.A), :L))
    L = copyltri(m.L)
    n, p, q, k = size(m)
    l = size(A, 2)
    if all(issubset.(inds, Ref(diagind(l, l))))
        Λ = Diagonal(1.0I(l))
        Λ̇ = map(x -> indmat(x, A), inds)
        L̇ = [zeros(T, size(A)) for _ in eachindex(inds)]
    else
        throw(ArgumentError("code not yet written"))
    end
    optsum = deepcopy(m.optsum)
    optsum.optimizer = :LD_MMA
    ToyModel(A, L, Λ, inds, Λ̇, L̇, (n=n, p=p, q=q, k=k, nθ=nθ(m)), optsum)
end

function indmat(inds, A)
    result = Diagonal(falses(LinearAlgebra.checksquare(A)))
    result[inds] .= true
    result
end

function Λinds(re::ReMat{T,1}, m::Integer, rcoffset::Integer) where {T}
    [range(rcoffset*(m + 1) + 1, step=m+1, length=length(re.levels))]
end

function Λinds(m::LinearMixedModel)
    rcoffset = 0
    ans = StepRange{Int,Int}[]
    n = size(m.L, 1)
    for trm in m.allterms
        if isa(trm, ReMat)
            append!(ans, Λinds(trm, n, rcoffset))
            rcoffset += trm.adjA.m
        end
    end
    ans
end

"""
    symmetrize!(A::Matrix)

Symmetrize `A` in place from its lower triangle by adding its transpose to it
"""
function symmetrize!(A::Matrix)
    n = LinearAlgebra.checksquare(A)
    for j in 1:n
        A[j, j] *= 2
        for i in (j+1):n
            A[i, j] = A[j, i] = A[i, j] + A[j, i]
        end
    end
    A
end

function setθ!(m::ToyModel{T}, v::Vector{T}) where {T}
    length(v) == m.dims.nθ || throw(DimensionMismatch("v must have length $(m.dims.nθ)"))
    Λ = m.Λ
    for (θj, indsj) in zip(v, m.inds)
        fill!(view(Λ, indsj), θj)
    end
    L = lmul!(adjoint(Λ), rmul!(copyto!(m.L, m.A), Λ))
    for j in 1:m.dims.q
        L[j, j] += 1
    end
    for (L̇, Λ̇) in zip(m.L̇, m.Λ̇)
        symmetrize!(lmul!(adjoint(Λ̇), rmul!(copyto!(L̇, m.A), Λ)))
    end
    chol_unblocked_and_fwd!(m.L̇, m.L)
    m
end

function objective(tm::ToyModel)
    L, n = tm.L, tm.dims.n
    dispsq = abs2(last(L)) / n
    2*sum(log(L[j]) for j in diagind(size(L, 1), tm.dims.q)) + n * (1 + log2π + log(dispsq))
end

function fg!(g::AbstractVector{T}, tm::ToyModel{T}) where {T}
    L, L̇, dims = tm.L, tm.L̇, tm.dims
    if length(L̇) ≠ length(g)
        throw(DimensionMismatch("length(g) = $(length(g)) ≠ $(length(L̇)) = length(L̇)"))
    end
    n, q = dims.n, dims.q
    dispsq = abs2(last(L)) / n
    l = size(L, 1)
    f = 2*sum(log(L[j]) for j in diagind(l, q)) + n * (1 + log2π + log(dispsq))
    for (i, ld) in enumerate(L̇)
        g[i] = 2*(sum(ld[j]/L[j] for j in diagind(l, q)) + n * last(ld) / last(L))
    end
    f
end

function fit!(tm::ToyModel{T}) where {T}
    optsum = tm.optsum
    opt = Opt(optsum)
    function obj(x, g)
        setθ!(tm, x)
        isempty(g) ? objective(tm) : fg!(g, tm)
    end
    NLopt.min_objective!(opt, obj)
    optsum.finitial = obj(optsum.initial, T[])
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    optsum.feval = opt.numevals
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    ret == :ROUNDOFF_LIMITED && @warn("NLopt was roundoff limited")
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end

    tm
end        
