"""
Convert sparse to dense if the proportion of nonzeros exceeds a threshold.
A no-op for other matrix types.
"""
function densify(S,threshold=0.3)
    issparse(S) || return S
    nnz(S)/(*(size(S)...)) > threshold || return S
    isbits(eltype(S)) && return full(S)
    nzs = nonzeros(S)
    nz1 = nzs[1]
    T = typeof(nz1)
    isa(nz1,Array) && isbits(eltype(nz1)) || error("Nonzeros must be a bitstype or an array of same")
    sz1 = size(nz1)
    all(x->typeof(x) == T && size(x) == sz1, nzs) || error("Inconsistent dimensions in array nonzeros")
    M,N = size(S)
    m,n = size(nz1,1),size(nz1,2) # this construction allows for nz1 to be a vector
    res = Array(eltype(nz1),M*m,N*n)
    rv = rowvals(S)
    for j in 1:size(S,2)
        jm1 = j - 1
        for k in nzrange(S,j)
            copy!(sub(res,(rv[k]-1)*m+(1:m),jm1*n+(1:n)),nzs[k])
        end
    end
    res
end
