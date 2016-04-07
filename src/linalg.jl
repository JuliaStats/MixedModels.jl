Base.LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T}, B) = Base.LinAlg.A_ldiv_B!(D, B)

function Base.LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    dd = D.diag
    bd = B.diag
    if length(dd) ≠ length(bd)
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    for j in eachindex(bd)
        bd[j] /= dd[j]
    end
    B
end

function Base.LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
    m, n = size(B)
    dd = D.diag
    if length(dd) ≠ m
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    nzv = nonzeros(B)
    rv = rowvals(B)
    for j in 1:n, k in nzrange(B,j)
        nzv[k] /= dd[rv[k]]
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(A::UpperTriangular{T,HBlkDiag{T}}, B::DenseMatrix{T})
    m, n = size(B)
    aa = A.data.arr
    r, s, k = size(aa)
    if m ≠ Compat.LinAlg.checksquare(A)
        throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    end
    scr = Array(T,(r,n))
    for i in 1:k
        bb = sub(B, (i - 1) * r + (1:r), :)
        copy!(bb, Base.LinAlg.Ac_ldiv_B!(UpperTriangular(sub(aa, :, :, i)), copy!(scr, bb)))
    end
    B
end

function rowlengths(L::LowerTriangular)
    ld = L.data
    [(sl = slice(ld, i, 1:i); sqrt(dot(sl, sl))) for i in 1:size(L, 1)]
end
