nlower{T<:Integer}(n::T) = (n * (n + 1)) >>> 1
nlower(A::UniformScaling) = 1
nlower(A::UniformSc) = nlower(A.λ)
nlower(A::LowerTriangular) = nlower(checksquare(A))

"""
    getθ!{T}(v::AbstractVector{T}, A::LowerTriangular{T, Matrix{T}})

Overwrite `v` with the elements of the lower triangle of `A` (column-major ordering)
"""
function getθ!{T<:AbstractFloat}(v::StridedVector{T}, A::LowerTriangular{T,Matrix{T}})
    Ad = A.data
    n = checksquare(Ad)
    if length(v) ≠ nlower(n)
        throw(DimensionMismatch("length(v) = $(length(v)) should be $(nlower(A))"))
    end
    k = 0
    for j = 1:n, i in j:n
        v[k += 1] = Ad[i, j]
    end
    v
end
getθ!{T<:AbstractFloat}(v::StridedVector{T}, A::UniformScLT{T}) = getθ!(v, A.λ)

function getθ!{T}(v::AbstractVector{T}, A::UniformScaling{T})
    if length(v) != 1
        throw(DimensionMismatch("v must be of length 1"))
    end
    v[1] = A.λ
    v
end

"""
    getθ(A::LowerTriangular{T, Matrix{T}})

Return a vector of the elements of the lower triangle of `A` (column-major ordering)
"""
getθ(A::UniformScaling) = [A.λ]
getθ{T<:AbstractFloat}(A::UniformScLT{T}) = getθ(A.λ)
getθ{T<:AbstractFloat}(A::LowerTriangular{T,Matrix{T}}) = getθ!(Array(T, nlower(A)), A)
getθ(A::AbstractVector) = mapreduce(getθ, vcat, A)

"""
    lowerbd{T}(A::LowerTriangular{T})

Return the vector of lower bounds on the parameters, `θ`.

These are the elements in the lower triangle in column-major ordering.
Diagonals have a lower bound of `0`.  Off-diagonals have a lower-bound of `-Inf`.
"""
lowerbd(v::AbstractVector) = mapreduce(lowerbd, vcat, v)
lowerbd(A::UniformSc) = lowerbd(A.λ)
function lowerbd{T}(A::LowerTriangular{T})
    n = checksquare(A)
    res = fill(convert(T, -Inf), nlower(A))
    k = -n
    for j in (n + 1):-1:2
        res[k += j] = zero(T)
    end
    res
end
lowerbd{T}(A::UniformScaling{T}) = zeros(T, (1,))

"""
    LT(A)

Create a uniform scaling, lower triangular, matrix compatible with the blocks of `A`
and initialized to the identity.
"""
LT{T}(A::ScalarReMat{T}) = UniformScaling(one(T))
function LT{T}(A::VectorReMat{T})
    k = size(A.z, 1)
    UniformSc(LowerTriangular((eye(T, k))))
end

function setθ!{T}(A::UniformScLT{T}, v::AbstractVector{T})
    Ad = A.λ.data
    n = checksquare(Ad)
    if length(v) ≠ nlower(n)
        throw(DimensionMismatch("length(v) = $(length(v)) should be $(nlower(A))"))
    end
    offset = 0
    for j in 1:n, i in j:n
        Ad[i, j] = v[offset += 1]
    end
    A
end

function setθ!{T}(A::UniformScaling{T}, v::AbstractVector{T})
    if length(v) != 1
        throw(DimensionMismatch("length(v) = $(length(v)) should be 1"))
    end
    A.λ = v[1]
    A
end
