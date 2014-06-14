## Utilities 

## function Base.Ac_mul_B!(α::Number, A::SparseMatrixCSC, B::AbstractMatrix, β::Number, C::AbstractMatrix)
##     mB, nB = size(B); A.m == mB || throw(DimensionMismatch(""))
##     for i = 1:length(C); C[i] *= β; end
##     for multivec_col = 1:nB
##          for col = 1 : A.n, k = A.colptr[col] : (A.colptr[col+1]-1)
##              C[col, multivec_col] += α*conj(A.nzval[k])*B[A.rowval[col], multivec_col]
##          end
##     end
##     C
## end

## function Base.At_mul_B!(α::Number, A::SparseMatrixCSC, B::AbstractMatrix, β::Number, C::AbstractMatrix)
##     mB, nB = size(B); A.m == mB || throw(DimensionMismatch(""))
##     for i = 1:length(C); C[i] *= β; end
##     for multivec_col = 1:nB
##          for col = 1 : A.n, k = A.colptr[col] : (A.colptr[col+1]-1)
##              C[col, multivec_col] += α*A.nzval[k]*B[A.rowval[k], multivec_col]
##          end
##     end
##     C
## end

## convert a lower Cholesky factor to a correlation matrix
function chol2cor(c::Matrix{Float64})
    m = Base.LinAlg.chksquare(c)
    m == 1 && return ones(1,1)
    std = broadcast(/, c, rowlengths(c))
    std * std'
end
chol2cor(t::Triangular) = chol2cor(full(t))

## ltri(m) -> v : extract the lower triangle as a vector
function ltri(m::Matrix)
    n = size(m,1)
    n == 1 && return copy(vec(m))
    res = Array(eltype(m),n*(n+1)>>1)
    pos = 0
    for j in 1:n, i in j:n
        res[pos += 1] = m[i,j]
    end
    res
end
ltri(t::Triangular{Float64,Array{Float64,2},:L,false}) = ltri(t.data)

## rowlengths(m) -> v : return a vector of the row lengths
rowlengths(m::Matrix{Float64}) = [norm(sub(m,i,:))::Float64 for i in 1:size(m,1)]
rowlengths(t::Triangular{Float64}) = rowlengths(full(t))

## Return a block in the Zt matrix from one term.
function ztblk(m::Matrix,v)
    nr,nc = size(m)
    nblk = maximum(v)
    NR = nr*nblk                        # number of rows in result
    cf = length(m) < typemax(Int32) ? int32 : int64 # conversion function
    SparseMatrixCSC(NR,nc,
                    cf(cumsum(vcat(1,fill(nr,(nc,))))), # colptr
                    cf(vec(reshape([1:NR],(nr,int(nblk)))[:,v])), # rowval
                    vec(m))            # nzval
end
ztblk(m::Matrix,v::PooledDataVector) = ztblk(m,v.refs)
