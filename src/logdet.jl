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
    r = sum(log(aa[i]) for i in dd)
    for j in 2:k
        dd += pq
        r += sum(log(aa[i]) for i in dd)
    end
    r
end

LD(d::DenseMatrix) = sum(i -> log(d[i]), diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` calculated in place.
"""
Base.logdet(m::LinearMixedModel) = 2 * sum(LD, view(diag(m.L), 1:length(m.Λ)))
