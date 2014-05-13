## Utilities 

function Base.Ac_mul_B!(α::Number, A::SparseMatrixCSC, B::AbstractMatrix, β::Number, C::AbstractMatrix)
    mB, nB = size(B); A.m == mB || throw(DimensionMismatch(""))
    for i = 1:length(C); C[i] *= β; end
    for multivec_col = 1:nB
         for col = 1 : A.n, k = A.colptr[col] : (A.colptr[col+1]-1)
             C[col, multivec_col] += α*conj(A.nzval[k])*B[A.rowval[col], multivec_col]
         end
    end
    C
end

function Base.At_mul_B!(α::Number, A::SparseMatrixCSC, B::AbstractMatrix, β::Number, C::AbstractMatrix)
    mB, nB = size(B); A.m == mB || throw(DimensionMismatch(""))
    for i = 1:length(C); C[i] *= β; end
    for multivec_col = 1:nB
         for col = 1 : A.n, k = A.colptr[col] : (A.colptr[col+1]-1)
             C[col, multivec_col] += α*A.nzval[k]*B[A.rowval[k], multivec_col]
         end
    end
    C
end

## convert a lower Cholesky factor to a correlation matrix
function cc(c::Matrix{Float64})
    m,n = size(c); m == n || error("argument of size $(size(c)) should be square")
    m == 1 && return ones(1,1)
    std = broadcast(/, c, Float64[norm(c[i,:]) for i in 1:size(c,1)])
    std * std'
end

## used in solve!(m::LMMGeneral, ubeta=false)
function cmult!(nzmat::Matrix,cc::StridedVecOrMat,inds::Vector,perm::Vector)
    for j in 1:size(cc,2)
        @inbounds for jj in 1:size(nzmat,2), i in 1:size(nzmat,1) scrm[i,jj] = nzmat[i,jj]*cc[jj,j] end
        @inbounds for i in 1:length(scrm) scrv[rvperm[i],j] += scrm[i] end
    end
    scrv
end

## return the vector of lower bounds for nonzeros in a lower
## triangular matrix of size k 
function lower_bd_ltri(k::Integer)
    pos = 0; res = zeros(k*(k+1)>>1)
    for j in 1:k
        pos += 1
        for i in (j+1):k
            res[pos += 1] = -Inf
        end
    end
    res
end

## ltri(M) -> vector of elements from the lower triangle (column major order)    
function ltri(M::Matrix)
    m,n = size(M); m == n || error("size(M) = ($m,$n), should be square")
    m == 1 && return vec(M)
    r = Array(eltype(M), m*(m+1)>>1); pos = 0
    for j in 1:m, i in j:m; r[pos += 1] = M[i,j]; end;
    r
end


## Is f nested within g, i.e does each value of f correspond to only one value of g?
function isnested(f::Vector,g::Vector)
    length(f) == length(g) || error("Dimension mismatch")
    uf = unique(f); ug = unique(g)
    isperm(uf) && isperm(ug) || error("unique(f) and unique(g) must be permutations")
    (nlf = length(uf)) >= (nlg = length(ug)) || error("f must have more levels than g")
    zz = zeros(eltype(g), nlf)
    for i in 1:length(g)
        if (z = zz[(fi = f[i])]) == (gi = g[i]) continue end
        if z == zero(eltype(g))
            zz[fi] = gi
        else
            return false
        end
    end
    true
end
