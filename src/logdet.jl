"""
`LD(A) -> log(det(triu(A)))` for `A` diagonal, HBlkDiag, or UpperTriangular
"""
function LD{T}(d::Diagonal{T})
    r = log(one(T))
    dd = d.diag
    for i in eachindex(dd)
        r += log(dd[i])
    end
    r
end

function LD{T}(d::HBlkDiag{T})
    r = log(one(T))
    aa = d.arr
    p,q,k = size(aa)
    for j in 1:k, i in 1:p
        r += log(aa[i,i,j])
    end
    r
end

function LD{T}(d::DenseMatrix{T})
    r = log(one(T))
    n = Compat.LinAlg.checksquare(d)
    for j in 1:n
        r += log(d[j,j])
    end
    r
end

"""
returns `log(det(Λ'Z'ZΛ + I))`
"""
function Base.logdet(m::LinearMixedModel)
    R = m.R
    s = 0.
    for i in eachindex(m.Λ)
        s += LD(R[i,i])
    end
    2.*s
end
