LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T}, B) = A_ldiv_B!(D, B)
function LinAlg.A_rdiv_Bc!{T}(A::StridedMatrix{T}, D::Diagonal{T})
    m,n = size(A)
    dd = D.diag
    if length(dd) ≠ n
        throw(DimensionMismatch("size(A, 2) = $n ≠ size(D, 2) = $(length(dd))"))
    end
    @inbounds for j in 1 : n
        ddj = dd[j]
        for i in 1 : m
            A[i, j] /= ddj
        end
    end
    A

end

function LinAlg.A_rdiv_Bc!{T}(A::SparseMatrixCSC{T}, D::Diagonal{T})
    m,n = size(A)
    dd = D.diag
    if length(dd) ≠ n
        throw(DimensionMismatch("size(A, 2) = $n ≠ size(D, 2) = $(length(dd))"))
    end
    nonz = nonzeros(A)
    for j in 1 : n
        ddj = dd[j]
        for k in nzrange(A, j)
            nonz[k] /= ddj
        end
    end
    A
end

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    if size(D) ≠ size(B)
        throw(DimensionMismatch("size(D) ≠ size(B)"))
    end
    map!(/, B.diag, B.diag, D.diag)
    B
end

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
    dd = D.diag
    if length(dd) ≠ size(B, 1)
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    vals = nonzeros(B)
    rows = rowvals(B)
    @inbounds for k in eachindex(vals)
        vals[k] /= dd[rows[k]]
    end
    B
end

function rowlengths(L::LowerTriangular)
    ld = L.data
    [norm(view(ld, i, 1:i)) for i in 1 : size(ld, 1)]
end

rowlengths(L::UniformScaling) = [abs(L.λ)]
