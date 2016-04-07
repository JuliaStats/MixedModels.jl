"""
    reweight!(m)

Incorporate `m.weights` in the products of terms in `m.A`

Args:

- `m`: a `MixedModel`

Returns:
  `m` with the products in `m.A` reweighted
"""
function reweight!(m::MixedModel)
    lm = lmm(m)
    A, trms, wts = lm.A, lm.trms, lm.weights
    if length(wts) ≠ size(trms[1], 1)
        throw(DimensionMismatch("$(length(wts)) = length(m.weights) ≠ size(m.trms[1], 1)"))
    end
    for j in 1:size(A, 2), i in 1:j
        wtprod!(A[i, j], trms[i], trms[j], wts)
    end
    m
end

function wtprod!{T <: AbstractFloat}(A::Diagonal{T}, ti::ScalarReMat{T}, tj::ScalarReMat{T}, wt::Vector{T})
    n, q = size(ti)
    if ti === tj
        ad = fill!(A.diag, zero(T))
        z = ti.z
        if length(ad) ≠ q || length(wt) ≠ n || length(z) != n
            throw(DimensionMismatch("size(A) should be $q and length(wt) should be $n"))
        end
        tir = ti.f.refs
        for i in eachindex(tir, wt)
            ad[tir[i]] += abs2(z[i]) * wt[i]
        end
        return A
    end
    error("Shouldn't happen?")
end

function wtprod!{T <: AbstractFloat}(A::Matrix{T}, ti::ScalarReMat{T}, tj::Matrix{T}, wt::Vector{T})
    r, m = size(ti)
    q, n = size(tj)
    if r ≠ q || length(wt) ≠ r || size(A, 1) ≠ m || size(A, 2) ≠ n
        throw(DimensionMismatch("matrix product dimensions"))
    end
    fill!(A, 0)
    z = ti.z
    tir = ti.f.refs
    for j in 1:n
        for k in eachindex(wt)
            A[tir[k], j] += wt[k] * z[k] * tj[k, j]
        end
    end
    A
end

function wtprod!{T <: AbstractFloat}(A::Matrix{T}, ti::Matrix{T}, tj::Matrix{T}, wt::Vector{T})
    Ac_mul_B!(A, ti, scale(wt, tj))
end

function wtprod!{T <: AbstractFloat}(A::SparseMatrixCSC{T}, ti::ScalarReMat{T}, tj::ScalarReMat{T}, wt::Vector{T})
    r, m = size(ti)
    q, n = size(tj)
    if r ≠ q || length(wt) ≠ r || size(A, 1) ≠ m || size(A, 2) ≠ n
        throw(DimensionMismatch("matrix product dimensions"))
    end
    nz, rv, cp = nonzeros(A), rowvals(A), A.colptr
    fill!(nz, 0)
    zi, zj, tir = ti.z, tj.z, ti.f.refs
    for j in 1:n
        ajind = nzrange(A, j)
        ajrv = sub(rv, ajind)
        for k in eachindex(wt)
            ff = searchsortedfirst(ajrv, tir[k])
            if ff > length(ajind)
                error(string("A[", tir[k], ", ", j, "] should be a nonzero"))
            end
            i = ajind[ff]
            nz[i] += wt[k] * zi[k] * zj[k]
        end
    end
    A
end
