abstract AbstractPDMatFactor
abstract SimplePDMatFactor <: AbstractPDMatFactor

if VERSION < v"0.4-"
    immutable PDLCholF <: SimplePDMatFactor
        ch::Cholesky{Float64}
    end
else
    immutable PDLCholF <: SimplePDMatFactor
        ch::Cholesky{Float64,Matrix{Float64},:L}
    end
end

type PDCompF <: AbstractPDMatFactor
    v::Vector{SimplePDMatFactor}
    t::Triangular
    PDCompF(v::Vector{SimplePDMatFactor}) = new(v,Triangular(eye(sum(dim,v)),:L,false))
end

immutable PDDiagF <: SimplePDMatFactor
    d::Diagonal{Float64}
    function PDDiagF(d::Vector{Float64})
        for dd in d
            dd < 0. && error("PDDiagF must have non-negative diagonal elements")
        end
        new(Diagonal(d))
    end
end

type PDScalF <: SimplePDMatFactor
    s::Float64
    n::Int
    function PDScalF(s::Number,n::Int)
        (s < 0. || n ≤ 0) && error("PDScalF n = $n,  scale, s = $s, must be non-negative")
        new(s,n)
    end
end

Base.copy!(m::StridedMatrix{Float64},p::PDLCholF) = tril!(copy!(m,p.ch.UL))

function Base.copy!(m::StridedMatrix{Float64},p::PDDiagF)
    n = dim(p)
    size(m) == (n,n) || throw(DimensionMismatch(""))
    for i in 1:n
        m[i,i] = p.d[i]
    end
end
function Base.copy!(m::StridedMatrix{Float64},p::PDScalF)
    size(m) == (p.n,p.n) || throw(DimensionMismatch(""))
    for i in 1:p.n
        m[i,i] = p.d[i]
    end
end

## full returns the covariance matrices
Base.full(p::PDCompF) = p.t * p.t'
Base.full(p::PDLCholF) = full(p.ch)
Base.full(p::PDDiagF) = full(Diagonal(abs2(p.d.diag)))
Base.full(p::PDScalF) = abs2(p.s)*eye(p.n)

Base.size(p::PDLCholF) = size(p.ch)
Base.size(p::PDLCholF,i) = size(p.ch,i)
Base.size(p::PDScalF) = (p.n,p.n)
function Base.size(p::PDScalF,i)
    i ≤ 0 && throw(BoundsError())
    1 ≤ i ≤ 2 && return p.n
    1
end
Base.size(p::PDCompF) = size(p.t)
Base.size(p::PDCompF,i) = size(p.t,i)
Base.size(p::PDDiagF) = (n=length(p.d);(n,n))

PDMats.dim(p::PDLCholF) = size(p.ch,1)
PDMats.dim(p::PDDiagF) = length(p.d.diag)
PDMats.dim(p::PDScalF) = p.n
PDMats.dim(p::PDCompF) = size(p.t,1)

Base.cond(p::PDLCholF) = cond(p.ch[:L])
Base.cond(p::PDDiagF) = ((m,M) = extrema(p.d.diag); M/m)
Base.cond(p::PDScalF) = 1.

nltri(k::Integer) = k*(k+1) >> 1

## number of free variables in the representation
nθ(p::PDCompF) = sum(nθ,p.v)
nθ(p::PDLCholF) = nltri(dim(p))
nθ(p::PDDiagF) = dim(p)
nθ(p::PDScalF) = 1

## current values of the free variables in the representation
function θ(p::PDLCholF)
    n = size(p,1)
    res = Array(Float64,nltri(n))
    pos = 0
    ul = p.ch.UL
    for j in 1:n, i in j:n
        res[pos += 1] = ul[i,j]
    end
    res
end
θ(p::PDCompF) = vcat(map(θ,p.v)...)
θ(p::PDDiagF) = p.d.diag
θ(p::PDScalF) = [p.s]

## lower bounds on the free variables in the representation
function lower(p::PDLCholF)
    n = size(p,1)
    res = fill(-Inf,nltri(n))
    i = 1                               # position in res
    for j in n:-1:1
        res[i] = 0.
        i += j
    end
    res
end
lower(p::PDCompF) = vcat(map(lower,p.v)...)
lower(p::PDDiagF) = zeros(p.d.diag)
lower(p::PDScalF) = [0.]

## θ!(m,theta) -> m : install new values of the free variables
function θ!(p::PDLCholF,th::StridedVector{Float64})
    n = size(p,1)
    length(th) == nθ(p) || throw(DimensionMismatch(""))
    pos = 0
    ul = p.ch.UL
    for j in 1:n, i in j:n
        ul[i,j] = th[pos += 1]
    end
    p
end

function θ!(p::PDDiagF,th::StridedVector{Float64})
    length(th) == length(p.d.diag) || throw(DimensionMisMatch(""))
    copy!(p.d.diag,th)
    p
end
function θ!(p::PDScalF,th::StridedVector{Float64})
    length(th) == 1 || throw(DimensionMisMatch(""))
    p.s = th[1]
    p
end
function θ!(p::PDCompF,th::StridedVector{Float64})
    nθv = map(nθ,p.v)
    sum(nθv) == length(th) || throw(DimensionMismatch(""))
    θoffset = 0
    coloffset = 0
    for i in 1:length(p.v)
        inds = coloffset + dim(p.v[i])
        coloffset += p.v[i]
        copy!(view(p.t.data,inds,inds),θ!(p.v[i],view(th,θoffset + (1:nθv[i]))))
        θoffset += nθv[i]
    end
    p
end

Base.A_mul_B!(A::PDCompF,B::StridedVecOrMat{Float64}) = A_mul_B!(A.t,B)
Base.A_mul_B!(A::PDDiagF,B::StridedVecOrMat{Float64}) = A_mul_B!(A.d,B)
Base.A_mul_B!(A::PDLCholF,B::StridedVecOrMat{Float64}) = A_mul_B!(A.ch[:L],B)
Base.A_mul_B!(A::PDScalF,B::StridedVecOrMat{Float64}) = scale!(A.s,B)

function Base.A_mul_B!{T<:Number}(A::StridedMatrix{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    length(dd) == n || throw(DimensionMismatch(""))
    @inbounds for j in 1:n
        dj = dd[j]
        for i in 1:m
            A[i,j] *= dj
        end
    end
    A
end

function Base.Ac_mul_B!{T<:Number}(A::Diagonal{T},B::StridedMatrix{T})
    m,n = size(B)
    dd = A.diag
    length(dd) == m || throw(DimensionMismatch(""))
    @inbounds for j in 1:n
        for i in 1:m
            B[i,j] *= conj(dd[i])
        end
    end
    B
end

Base.A_mul_B!(A::StridedMatrix{Float64},B::PDCompF) = A_mul_B!(A,B.t)
Base.A_mul_B!(A::StridedMatrix{Float64},B::PDDiagF) = A_mul_B!(A,B.d)
Base.A_mul_B!(A::StridedMatrix{Float64},B::PDLCholF) = A_mul_B!(A,B.ch[:L])
Base.A_mul_B!(A::StridedMatrix{Float64},B::PDScalF) = scale!(A,B.s)

Base.A_mul_Bc!(A::StridedMatrix{Float64},B::PDCompF) = A_mul_Bc!(A,B.t)
Base.A_mul_Bc!(A::StridedMatrix{Float64},B::PDDiagF) = A_mul_Bc!(A,B.d)
Base.A_mul_Bc!(A::StridedMatrix{Float64},B::PDLCholF) = A_mul_Bc!(A,B.ch[:L])
Base.A_mul_Bc!(A::StridedMatrix{Float64},B::PDScalF) = scale!(A,conj(B.s))

Base.Ac_mul_B!(A::PDCompF,B::StridedVecOrMat{Float64}) = Ac_mul_B!(A.t,B)
Base.Ac_mul_B!(A::PDLCholF,B::StridedVecOrMat{Float64}) = Ac_mul_B!(A.ch[:L],B)
Base.Ac_mul_B!(A::PDDiagF,B::StridedVecOrMat{Float64}) = Ac_mul_B!(A.d,B)
Base.Ac_mul_B!(A::PDScalF,B::StridedVecOrMat{Float64}) = scale!(conj(A.s),B)

rowlengths(p::PDCompF) = vcat(map(rowlengths,p.v)...)
function rowlengths(p::PDLCholF)
    k = dim(p)
    ul = p.ch[:L].data
    [norm(view(ul,i,1:i)) for i in 1:k]
end
rowlengths(p::PDScalF) = fill(p.s,(p.n,))
rowlengths(p::PDDiagF) = p.d.diag

function chol2cor(p::Union(PDCompF,PDLCholF))
    (m = dim(p)) == 1 && return ones(1,1)
    res = full(p.ch)
    d = [inv(sqrt(dd)) for dd in diag(res)]
    scale!(d,scale!(res,d))
end
chol2cor(p::PDDiagF) = eye(p.d)
chol2cor(p::PDScalF) = eye(p.n)

Base.svdvals(p::PDLCholF) = svdvals(p.ch[:L])
Base.svdvals(p::PDDiagF) = convert(Vector{Float64},sort(p.d.diag; rev=true))
Base.svdvals(p::PDScalF) = fill(p.s,(p.n,))

Base.svdfact(p::PDLCholF) = svdfact(p.ch[:L])

## FIXME make this work
function grdcmp!(v::DenseVector{Float64},p::PDCompF,m::Matrix{Float64})
    pv = map(dim,p.v)
    (n = chksquare(m)) == sum(pv) || throw(DimensionMismatch(""))
    coloffset = 0
    for i in 1:length(pv)
        inds = coloffset + (1:pv[i])
        coloffset += pv[i]
        push!(v,grdcmp(view(m,inds,inds),p.v[i]))
    end
    v
end
    
function grdcmp!(v::DenseVector{Float64},p::PDLCholF,m::Matrix{Float64})
    (n = chksquare(m)) == dim(p) || throw(DimensionMismatch(""))
    length(v) == nltri(n) || throw(DimensionMismatch(""))
    pos = 0
    for i in 1:n, j in i:n
        v[pos += 1] = 2.m[i,j]
    end
    v
end

function grdcmp!(v::DenseVector{Float64},p::PDDiagF,m::Matrix{Float64})
    (n = dim(p)) == chksquare(m) == length(v) || throw(DimensionMismatch(""))
    for i in 1:n
        v[i] = 2.m[i,i]
    end
    v
end

function grdcmp!(v::DenseVector{Float64},p::PDScalF,m::Matrix{Float64})
    (n = chksquare(m)) == dim(p) || throw(DimensionMismatch(""))
    length(v) == 1 || throw(DimensionMismatch(""))
    v[1] = 2.(n == 1 ? m[1,1] : sum(diag(m)))
end

Base.tril(p::PDLCholF) = tril(p.ch.UL)
Base.tril(p::PDDiagF) = diagm(p.d)
Base.tril(p::PDScalF) = diagm(fill(p.s,(p.n,)))

function amalgamate1(Xs,p,λ)
    (k = length(λ)) == length(Xs) == length(p) || throw(DimensionMismatch(""))
    k == 1 && return (Xs,p,λ)
    if all([isa(ll,PDScalF) for ll in λ])
        return({vcat(Xs...)},[sum(p)],{PDDiagF(ones(length(λ)))},)
    end
    error("Composite code not yet written")
end

## amalgamate random-effects terms with identical grouping factors
function amalgamate(grps,Xs,p,λ,facs,l)
    np = Int[]; nXs = {}; nλ = {}; nfacs = {}; nl = Int[]
    ugrp = unique(grps)
    for u in ugrp
        inds = grps .== u
        (xv,pv,lv) = amalgamate1(Xs[inds],p[inds],λ[inds])
        append!(np,pv)
        append!(nXs,xv)
        append!(nλ,lv)
        append!(nfacs,{facs[inds[1]]})
        push!(nl,l[inds[1]])
    end
    ugrp,nXs,np,nλ,nfacs,nl
end
