"""
`ReMat` - model matrix for a random-effects term
"""
abstract ReMat

"""
`ScalarReMat` - a model matrix for scalar random effects

The matrix is represented by the grouping factor, `f`, and a vector `z`.
"""
immutable ScalarReMat{T} <: ReMat
    f::PooledDataVector
    z::Vector{T}
    fnm::Symbol
    cnms::Vector
end

"""
`VectorReMat` - a representation of a model matrix for vector-valued random effects

The matrix is represented by the grouping factor, `f`, and the transposed raw
model matrix, `z`.
"""
immutable VectorReMat{T} <: ReMat
    f::PooledDataVector                 # grouping factor
    z::Matrix{T}
    fnm::Symbol
    cnms::Vector
end

"""
`remat(e,df)` -> `ReMat`

A factory for `ReMat` objects constructed from a random-effects term and a
`DataFrame`
"""
function remat(e::Expr,df::DataFrame)
    e.args[1] == :| || throw(ArgumentError("$e is not a call to '|'"))
    fnm = e.args[3]
    gr = getindex(df,fnm)
    gr = isa(gr,PooledDataArray) ? gr : pool(gr)
    if e.args[2] == 1
        return ScalarReMat(gr,ones(length(gr)),fnm,["(Intercept)"])
    end
    mf = ModelFrame(Formula(nothing,e.args[2]),df)
    z = ModelMatrix(mf).m
    cnms = coefnames(mf)
    size(z,2) == 1 ?ScalarReMat(gr,vec(z),fnm,cnms) : VectorReMat(gr,z',fnm,cnms)
end

Base.eltype(R::ReMat) = eltype(R.z)  # not tested

vsize(A::ReMat) = isa(A,ScalarReMat) ? 1 : size(A.z,1)

Base.size(A::ReMat) = (length(A.f),vsize(A)*length(A.f.pool))

Base.size(A::ReMat,i::Integer) =
    i < 1 ? throw(BoundsError()) :
    i == 1 ? length(A.f) :
    i == 2 ? vsize(A)*length(A.f.pool) : 1


==(A::ReMat,B::ReMat) = (A.f == B.f) && (A.z == B.z)

# FIXME add a tA boolean argument to combine the code for A_mul_B! and Ac_mul_B!
function Base.A_mul_B!{T}(α::Real,A::ReMat,B::StridedVecOrMat{T},β::Real,R::StridedVecOrMat{T}) # not tested
    n,q = size(A)
    k = size(B,2)
    size(R,1) == n && size(B,1) == q && size(R,2) == k || throw(DimensionMismatch())
    if β ≠ 1
        scale!(β,R)
    end
    rr = A.f.refs
    zz = A.z
    if isa(A,ScalarReMat)
        for j in 1:k, i in 1:n
            R[i,j] += α * zz[i] * B[rr[i],j]
        end
    else
        l = size(zz,1)
        Bt = reshape(B,(l,div(q,l),k))
        for j in 1:k, i in 1:n
            R[i,j] += α * dot(sub(Bt,:,Int(rr[i]),j),sub(zz,:,i))
        end
    end
    R
end

# not tested
Base.A_mul_B!{T}(A::ReMat,B::StridedVecOrMat{T},R::StridedVecOrMat{T}) = A_mul_B!(one(T),A,B,zero(T),R)

function Base.Ac_mul_B!{T}(α::Real,A::ReMat,B::StridedVecOrMat{T},β::Real,R::StridedVecOrMat{T})
    n,q = size(A)
    k = size(B,2)
    size(R,1) == q && size(B,1) == n && size(R,2) == k || throw(DimensionMismatch())
    if β ≠ 1
        scale!(β,R)
    end
    rr = A.f.refs
    zz = A.z
    if isa(A,ScalarReMat)
        for j in 1:k, i in 1:n
            R[rr[i],j] += α * zz[i] * B[i,j]
        end
    else
        l = size(zz,1)
        rt = reshape(R,(l,div(q,l),k))
        for j in 1:k, i in 1:n
            Base.axpy!(α*B[i,j],sub(zz,:,i),sub(rt,:,Int(rr[i]),j))
        end
    end
    R
end

Base.Ac_mul_B!{T}(R::StridedVecOrMat{T},A::ReMat,B::StridedVecOrMat{T}) = Ac_mul_B!(one(T),A,B,zero(T),R)

function Base.Ac_mul_B(A::ReMat,B::DenseVecOrMat)
    k = size(A,2)
    Ac_mul_B!(zeros(eltype(B), isa(B,Vector) ? (k,) : (k, size(B,2))), A, B)
end

function Base.Ac_mul_B(A::ScalarReMat, B::ScalarReMat)
    Az = A.z
    Ar = A.f.refs
    if is(A,B)
        v = zeros(eltype(Az),length(A.f.pool))
        for i in eachindex(Ar)
            v[Ar[i]] += abs2(Az[i])
        end
        return Diagonal(v)
    end
    densify(sparse(convert(Vector{Int32},Ar),convert(Vector{Int32},B.f.refs),Az .* B.z))
end

function Base.Ac_mul_B(A::VectorReMat,B::VectorReMat)
    Az = A.z
    Ar = convert(Vector{Int},A.f.refs)
    if is(A,B)
        l,n = size(Az)
        T = eltype(Az)
        np = length(A.f.pool)
        a = zeros(T,(l,l,np))
        for i in eachindex(Ar)
            Base.LinAlg.BLAS.syr!('L',one(T),sub(Az,:,i),sub(a,:,:,Ar[i]))
        end
        for k in 1:np
            Base.LinAlg.copytri!(sub(a,:,:,k),'L')
        end
        return HBlkDiag(a)
    end
    Bz = B.z
    Br = convert(Vector{Int},B.f.refs)
    (m = length(Ar)) == length(Br) || throw(DimensionMismatch())
    sparse(Ar,Br,[sub(Az,:,i)*sub(Bz,:,i)' for i in 1:m])
end

function Base.Ac_mul_B!{T}(R::DenseVecOrMat{T},A::DenseVecOrMat{T},B::ReMat)
    m = size(A,1)
    n = size(A,2)
    p,q = size(B)
    m == p && size(R,1) == n && size(R,2) == q || throw(DimensionMismatch(""))
    fill!(R,zero(T))
    rr = B.f.refs
    zz = B.z
    if isa(B,ScalarReMat)
        for j in 1:n, i in 1:m
            R[j,rr[i]] += A[i,j] * zz[i]
        end
    else # branch not tested
        l = size(zz,1)
        for j in 1:n, i in 1:m
            Base.LinAlg.axpy!(A[i,j],sub(zz,:,i),sub(R,j,(rr[i]-1)*l + (1:l)))
        end
    end
    R
end

# not tested
Base.Ac_mul_B(A::DenseVecOrMat,B::ReMat) = Ac_mul_B!(Array(eltype(A),(size(A,2),size(B,2))),A,B)

# not tested
function Base.LinAlg.scale{T}(d::Vector{T},A::ScalarReMat{T})
    ScalarReMat(A.f, d .* copy(A.z),A.fnm,A.cnms)
end

#not tested
function Base.LinAlg.scale{T}(d::Vector{T},A::VectorReMat{T})
    VectorReMat(A.f,scale(A.z,d),A.fnm,A.cnms)
end
