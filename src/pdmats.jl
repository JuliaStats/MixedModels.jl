abstract AbstractPDMatFactor

immutable PDLCholF <: AbstractPDMatFactor
    ch::Cholesky{Float64}
    function PDLCholF(ch::Cholesky{Float64})
        ch.uplo == 'L' || error("PDLCholF must be a lower Cholesky factor")
        new(ch)
    end
end
immutable PDDiagF <: AbstractPDMatFactor
    d::Diagonal{Float64}
    function PDDiagF(d::Vector{Float64})
        for dd in d
            dd < 0. && error("PDDiagF must have non-negative diagonal elements")
        end
        new(Diagonal(d))
    end
end
type PDScalF <: AbstractPDMatFactor     #  will need to change this to a dimension and scale
    s::UniformScaling{Float64}
    function PDScalF(s::Number)
        s < 0. && error("PDScalF scale, s = $s, must be non-negative")
        new(UniformScaling(float(s)))
    end
end

Base.show(io::IO,p::PDLCholF) = show(io,p.ch)
Base.show(io::IO,p::PDDiagF) = show(io,p.d)
Base.show(io::IO,p::PDScalF) = show(io,p.s)

## full returns the covariance matrices
Base.full(p::PDLCholF) = full(p.ch)
Base.full(p::PDDiagF) = full(Diagonal(abs2(p.d.diag)))
Base.full(p::PDScalF) = [abs2(p.s.λ)]

Base.size(p::PDLCholF) = size(p.ch)
Base.size(p::PDLCholF,i) = size(p.ch,i)

PDMats.dim(p::PDLCholF) = size(p.ch,1)
PDMats.dim(p::PDDiagF) = length(p.d.diag)
PDMats.dim(p::PDScalF) = -1

Base.cond(p::PDLCholF) = cond(p.ch[:L])
Base.cond(p::PDDiagF) = ((m,M) = extrema(p.d.diag); M/m)
Base.cond(p::PDScalF) = 1.

nltri(k::Integer) = k*(k+1) >> 1

## number of free variables in the representation
nθ(p::PDLCholF) = nltri(size(p,1))
nθ(p::PDDiagF) = size(p,1)
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
θ(p::PDDiagF) = p.d.diag
θ(p::PDScalF) = [p.s.λ]

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
end
function θ!(p::PDDiagF,th::StridedVector{Float64})
    length(th) == length(p.d.diag) || throw(DimensionMisMatch(""))
    copy!(p.d.diag,th)
end
function θ!(p::PDScalF,th::StridedVector{Float64})
    length(th) == 1 || throw(DimensionMisMatch(""))
    p.s = UniformScaling(th[1])
end

Base.A_mul_B!(A::PDLCholF,B::StridedVecOrMat{Float64}) = A_mul_B!(A.ch[:L],B)
Base.A_mul_B!(A::PDDiagF,B::StridedVecOrMat{Float64}) = A_mul_B!(A.d,B)
Base.A_mul_B!(A::PDScalF,B::StridedVecOrMat{Float64}) = scale!(A.s.λ,B)
Base.A_mul_B!(A::StridedVecOrMat{Float64},B::PDLCholF) = A_mul_B!(A,B.ch[:L])
Base.A_mul_B!(A::StridedVecOrMat{Float64},B::PDDiagF) = A_mul_B!(A,B.d)
Base.A_mul_B!(A::StridedVecOrMat{Float64},B::PDScalF) = scale!(A,B.s.λ)

Base.Ac_mul_B!(A::PDLCholF,B::StridedVecOrMat{Float64}) = Ac_mul_B!(A.ch[:L],B)
Base.Ac_mul_B!(A::PDDiagF,B::StridedVecOrMat{Float64}) = A_mul_B!(A.d,B)
Base.Ac_mul_B!(A::PDScalF,B::StridedVecOrMat{Float64}) = scale!(A.s.λ,B)

function rowlengths(p::PDLCholF)
    k = dim(p)
    ul = p.ch[:L].data
    [norm(view(ul,i,1:i)) for i in 1:k]
end
rowlengths(p::PDScalF) = [p.s.λ]
rowlengths(p::PDDiagF) = p.d.diag

function chol2cor(p::PDLCholF)
    (m = dim(p)) == 1 && return ones(1,1)
    res = full(p.ch)
    d = [inv(sqrt(dd)) for dd in diag(res)]
    scale!(d,scale!(res,d))
end
chol2cor(p::PDDiagF) = eye(p.d)
chol2cor(p::PDScalF) = eye(1)

Base.svdvals(p::PDLCholF) = svdvals(p.ch[:L])
Base.svdvals(p::PDDiagF) = sort(p.d.diag)
Base.svdvals(p::PDScalF) = [p.s.λ]

Base.svdfact(p::PDLCholF) = svdfact(p.ch[:L])
