"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return the value of `log(det(triu(A)))` calculated in place.
"""
LD(d::Diagonal) = sum(log, d.diag)

function LD(d::HBlkDiag)
    aa = d.arr
    p, q, k = size(aa)
    pq = p * q
    dd = diagind(p, q)
    r = sum(i -> log(aa[i]), dd)
    for j in 1:(k - 1)
        r += sum(i -> log(aa[i]), dd + j * pq)
    end
    r
end

LD(d::DenseMatrix) = sum(i -> log(d[i]), diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` calculated in place.
"""
function Base.logdet{T}(m::LinearMixedModel{T})
    k, R = length(m.Λ), m.R
    res = zero(T)
    for i in 1 : length(m.Λ)
        res += T(LD(R[i, i]))
    end
    2 * res
end
