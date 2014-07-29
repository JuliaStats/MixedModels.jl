function θ(t::Triangular{Float64,Array{Float64,2},:L,false})
    n = size(t,1)
    n == 1 && return copy(vec(t.data))
    res = Array(Float64,n*(n+1)>>1)
    pos = 0
    for j in 1:n, i in j:n
        res[pos += 1] = t.data[i,j]
    end
    res
end
θ(c::Cholesky{Float64}) = θ(c[:L])
θ(p::PDMat) = θ(p.chol)
θ(p::PDiagMat) = sqrt(p.diag)

nθ(k::Integer) = k*(k+1)>>1
nθ(t::Triangular{Float64,Array{Float64,2},:L,false}) = nθ(size(t,1))
nθ(c::Cholesky{Float64}) = nθ(size(c,1))
nθ(p::PDMat) = nθ(dim(p))
nθ(p::PDiagMat) = dim(p)

function lower(k::Integer)
    k == 1 && return [0.]
    res = fill(-Inf,k*(k+1)>>1)
    i = 1                               # position in res
    for j in k:-1:1
        res[i] = 0.
        i += j
    end
    res
end
lower(t::Triangular{Float64,Array{Float64,2},:L,false}) = lower(size(t,1))
lower(c::Cholesky{Float64}) = lower(size(c,1))
lower(p::PDMat) = lower(dim(p))
lower(p::PDiagMat) = zeros(dim(p))

## θ!(m,theta) -> m : install new values of the covariance parameters
function θ!(t::Triangular{Float64,Array{Float64,2},:L,false},th::StridedVector{Float64})
    k = size(t,1)
    length(th) == nθ(k) || throw(DimensionMismatch(""))
    pos = 0
    for j in 1:k, i in j:k
        t.data[i,j] = th[pos += 1]
    end
end
function θ!(c::Cholesky{Float64},th::StridedVector{Float64})
    c.uplo == 'L' || error("θ! defined for lower Cholesky factor only")
    θ!(c[:L],th)
end
function θ!(p::PDMat,th::StridedVector{Float64})
    θ!(p.chol,th)
    tl = p.chol[:L]
    A_mul_B!(tl,Base.LinAlg.transpose!(p.mat,tril!(tl.data)))
end
function θ!(p::PDiagMat,th::StridedVector{Float64})
    length(th) == p.dim || throw(DimensionMismatch(""))
    map!(abs2,p.diag,th)
    map!(inv,p.inv_diag,p.diag)
end

lfactor(p::PDMat) = p.chol[:L]
lfactor(p::PDiagMat) = Diagonal(sqrt(p.diag))

facdiag(p::PDMat) = diag(p.chol.UL)
facdiag(p::PDiagMat) = sqrt(p.diag)

Base.A_mul_B!{T<:Union(Float32,Float64)}(x::StridedMatrix{T},d::Diagonal{T}) = scale!(x,d.diag)

function unwhiten_sym!(p::PDMat,x::StridedMatrix)
    if p.chol.uplo == 'U'
        tu = p.chol[:U]
        Ac_mul_B!(tu,A_mul_B!(x,tu))
    else
        tl = p.chol[:L]
        A_mul_B!(tl,Ac_mul_B!(x,tl))
    end
end
function unwhiten_sym!(p::PDiagMat,x::StridedMatrix)
    (m = Base.LinAlg.chksquare(x)) == length(p.diag) || throw(DimensionMismatch(""))
    dfac = sqrt(p.diag)
    for j in 1:m, i in 1:m
        m[i,j] *= dfac[i]*dfac[j]
    end
end
