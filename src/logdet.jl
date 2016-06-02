"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)
The value of `log(det(triu(A)))` calculated in place.
"""
LD(d::Diagonal) = sum(log, d.diag)

function LD{T}(d::HBlkDiag{T})
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
The value of `log(det(Λ'Z'ZΛ + I))` calculated in place.
"""
function Base.logdet{T}(m::LinearMixedModel{T})
    R = sub(m.R, :, 1 : length(m.Λ))
    2. * T(sum(i -> LD(R[i]), diagind(R)))
end
