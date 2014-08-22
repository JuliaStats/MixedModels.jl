## Solver for models with two crossed or nearly-crossed grouping factors for the random effects

## There are l₁ and l₂ levels in the grouping factors.  The dimension
## of the random effects for each level of the grouping factors is p₁
## and p₂, respectively. The total number of random effects is q₁+q₂
## where qᵢ=pᵢlᵢ, i=1,2.  The dimension of the fixed-effects parameter
## is p.  When solving for the conditional modes of U only the
## transposed model matrix Xt passed to the constructor has 0 rows.

## Within the type:
## Z₁'Z₁ is block diagonal with l₁ blocks of size p₁×p₁, stored as A₁₁, a p₁×q₁ matrix
## Z₂'Z₁ is stored as A₂₁, a q₂×q₁ matrix.  The fraction of zeros in A₂₁ should be small.
## X'Z₁ is stored as A₃₁, a p×q₁ matrix.
## Z₂'Z₂ is block diagonal with l₂ blocks of size p₂×p₂, stored as A₂₂, a p₁×q₁ matrix
## X'Z₂ is stored as A₃₂, a p×q₂ matrix.
## X'X is stored as A₃₃, a p×p matrix - symmetric but not explicitly a Symmetric type
## L₁₁ is block diagonal with l₁ lower triangular blocks of size p₁×p₁, stored as a p₂×q₂ matrix
## L₂₁ is stored as a q₂×q₁ matrix
## L₃₁ is stored as a p×q₁ matrix
## L₂₂ is stored as a lower Triangular q₂×q₂ matrix, due to fill-in
## L₃₂ is stored as a p×q₂ matrix
## L₃₃ is stored as a p×p lower Cholesky factor

type PLSTwo <: PLSSolver # Solver for models with two crossed or nearly crossed terms
    A₁₁::Matrix{Float64} # diagonal blocks of Z₁'Z₁
    A₂₁::Matrix{Float64} # Z₂'*Z₁
    A₃₁::Matrix{Float64} # X'Z₁
    A₂₂::Matrix{Float64} # diagonal blocks of Z₂'Z₂
    A₃₂::Matrix{Float64} # X'Z₂
    A₃₃::Matrix{Float64} # X'X
    L₁₁::Matrix{Float64} 
    L₂₁::Matrix{Float64}
    L₃₁::Matrix{Float64}
    L₂₂::Triangular{Float64,Matrix{Float64},:L,false} 
    L₃₂::Matrix{Float64}
    L₃₃::Cholesky{Float64}
end

function PLSTwo(facs::Vector,Xst::Vector,Xt::Matrix)
    length(facs) == length(Xst) == 2 || throw(DimensionMismatch("PLSTwo"))
                                        # check for consistency in number of observations
    (n = size(Xt,2)) == size(Xst[1],2) == size(Xst[2],2) ==
        length(facs[1]) == length(facs[2]) || throw(DimensionMismatch(""))
    l₁ = length(facs[1].pool)
    l₂ = length(facs[2].pool)
    p₁ = size(Xst[1],1)
    p₂ = size(Xst[2],1)
    q₁ = p₁ * l₁
    q₂ = p₂ * l₂
    p = size(Xt,1)
                                        # Do this in lmm instead of checking here
    q₁ ≥ q₂ || error("reverse the order of the random effects terms")
    A₁₁ = zeros(p₁,q₁)
    A₂₁ = zeros(q₂,q₁)
    A₃₁ = zeros(p,q₁)
    A₂₂ = zeros(p₂,q₂)
    A₃₂ = zeros(p,q₂)
    r₁ = facs[1].refs
    r₂ = facs[2].refs
    for j in 1:n
        i₁ = r₁[j] - 1
        i₂ = r₂[j] - 1
        c₁ = contiguous_view(Xst[1],p₁*(j-1),(p₁,))
        c₂ = contiguous_view(Xst[2],p₂*(j-1),(p₂,))
        c₃ = contiguous_view(Xt,p*(j-1),(p,))
        BLAS.syr!('L',1.,c₁,contiguous_view(A₁₁,i₁*p₁*p₁,(p₁,p₁)))
        BLAS.ger!(1.,c₂,c₁,view(A₂₁,i₂*p₂+(1:p₂),i₁*p₁+(1:p₁)))
        BLAS.ger!(1.,c₃,c₁,contiguous_view(A₃₁,i₁*p*p₁,(p,p₁)))
        BLAS.syr!('L',1.,c₂,contiguous_view(A₂₂,i₂*p₂*p₂,(p₂,p₂)))
        BLAS.ger!(1.,c₃,c₂,contiguous_view(A₃₂,i₂*p*p₂,(p,p₂)))
    end
    PLSTwo(A₁₁,A₂₁,A₃₁,A₂₂,A₃₂,Xt*Xt',similar(A₁₁),similar(A₂₁),similar(A₃₁),
           Triangular(zeros(q₂,q₂),:L,false),similar(A₃₂),cholfact!(eye(p),:L))
end

function Base.cholfact(s::PLSTwo,RX::Bool=true)
    RX && return s.L₃₃
    p,p₁,p₂,l₁,l₂ = size(s)
    L₁₁ = blkdiag({sparse(tril(contiguous_view(s.L₁₁,j*p₁*p₁,(p₁,p₁)))) for j in 0:(l₁-1)}...)
    vcat(hcat(L₁₁,spzeros(p₁*l₁,p₂*l₂)),sparse(hcat(s.L₂₁,s.L₂₂)))
end

function Base.logdet(s::PLSTwo,RX=true)
    RX && return logdet(s.L₃₃)
    p,p₁,p₂,l₁,l₂ = size(s)
    sm = 0.
    for j in 0:(l₁-1), i in 1:p₁
        sm += log(s.L₁₁[i,j*p₁+i])
    end
    for i in 1:size(s.L₂₂,1)
        sm += log(s.L₂₂[i,i])
    end
    2.sm
end

function Base.A_ldiv_B!(s::PLSTwo,uβ)
    p,p₁,p₂,l₁,l₂,q₁,q₂ = size(s)
    length(uβ) == p+q₁+q₂ || throw(DimensionMismatch(""))
    u₁ = contiguous_view(uβ,(q₁,))
    u₂ = contiguous_view(uβ,q₁,(q₂,))    
    β = contiguous_view(uβ,q₁+q₂,(p,))
    if p₁ == 1                          # scalar r.e. for factor 1
        scaleinv!(u₁,vec(s.L₁₁))
    else
        for j in 0:(l₁-1)               # solve L₁ cᵤ₁ = λ₁'Z₁'y blockwise
            A_ldiv_B!(Triangular(contiguous_view(s.L₁₁,j*p₁*p₁,(p₁,p₁)),:L,false),
                      contiguous_view(uβ,j*p₁,(p₁,)))
        end
    end
    A_ldiv_B!(s.L₂₂,BLAS.gemv!('N',-1.,s.L₂₁,u₁,1.,u₂))
    A_ldiv_B!(s.L₃₃,BLAS.gemv!('N',-1.,s.L₃₂,u₂,1.,BLAS.gemv!('N',-1.,s.L₃₁,u₁,1.,β)))
    Ac_ldiv_B!(s.L₂₂,BLAS.gemv!('T',-1.,s.L₃₂,β,1.,u₂))
    BLAS.gemv!('T',-1.,s.L₃₁,β,1.,BLAS.gemv!('T',-1.,s.L₂₁,u₂,1.,u₁))
    if p₁ == 1
        scaleinv!(u₁,vec(s.L₁₁))
    else
        for j in 0:(l₁-1)                    # solve L₁₁'u₁ = cᵤ₁ blockwise
            Ac_ldiv_B!(Triangular(contiguous_view(s.L₁₁,j*p₁*p₁,(p₁,p₁)),:L,false),
                       contiguous_view(uβ,j*p₁,(p₁,)))
        end
    end
    uβ
end

function Base.size(s::PLSTwo)
    p₁,q₁ = size(s.A₁₁)
    q₁ % p₁ == 0 || throw(DimensionMismatch(""))
    p₂,q₂ = size(s.A₂₂)
    q₂ % p₂ == 0 || throw(DimensionMismatch(""))
    size(s.A₃₃,1),p₁,p₂,div(q₁,p₁),div(q₂,p₂),q₁,q₂
end

function update!(s::PLSTwo,λ::Vector)
    length(λ) == 2 || throw(DimensionMismatch(""))
    for ll in λ
        isa(ll,AbstractPDMatFactor) || error("λ must be a vector of PDMatFactors")
    end
    λ₁ = λ[1]
    λ₂ = λ[2]    
    p,p₁,p₂,l₁,l₂,q₁,q₂ = size(s)
    dim(λ₁) in [-1,p₁] || throw(DimensionMismatch(""))
    dim(λ₂) in [-1,p₂] || throw(DimensionMismatch(""))
    L₂₁ = copy!(s.L₂₁,s.A₂₁)
    L₃₁ = copy!(s.L₃₁,s.A₃₁)
    if p₁ == 1                          # shortcut for 1×1 λ
        isa(λ[1],PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ[1].s.λ
        lamsq = lam*lam
        for j in 1:l₁
            sc = lam/(s.L₁₁[1,j] = sqrt(s.A₁₁[1,j]*lamsq + 1.))
            scale!(contiguous_view(L₂₁,(j-1)*q₂,(q₂,)),sc)
            scale!(contiguous_view(L₃₁,(j-1)*p,(p,)),sc)
        end
    else
        Ac_mul_B!(λ₁,copy!(s.L₁₁,s.A₁₁)) # multiply on left by λ₁'
        for k in 0:(l₁-1)                # using offsets, not indices
            wL = A_mul_B!(contiguous_view(s.L₁₁,k*p₁*p₁,(p₁,p₁)),λ₁)
            for j in 1:p₁               # inflate diagonal
                wL[j,j] += 1.0
                for i in 1:(j-1)
                    wL[i,j] = 0.
                end
            end
            _,info = LAPACK.potrf!('L',wL) # lower Cholesky factor
            info == 0 || error("Cholesky failure at L₁₁ block $(k+1)")
            tr = Triangular(wL,:L,false)
            A_rdiv_Bc!(A_mul_B!(contiguous_view(L₂₁,k*q₂*p₁,(q₂,p₁)),λ₁),tr)
            A_rdiv_Bc!(A_mul_B!(contiguous_view(L₃₁,k*p*p₁,(p,p₁)),λ₁),tr)
        end
    end
                                        # second level updates
    L₂₂d = fill!(s.L₂₂.data,0.)
    L₃₂ = copy!(s.L₃₂,s.A₃₂)
    if p₂ == 1
        isa(λ₂,PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ₂.s.λ
        for i in 1:l₂
            L₂₂d[i,i] = s.A₂₂[1,i]*lam*lam + 1.
        end
        scale!(L₂₁,lam)
        scale!(L₃₂,lam)
    else
        Ac_mul_B!(λ₂,reshape(L₂₁,(p₂,l₂*q₁)))
        for k in 0:(l₂-1)
            kk = k*p₂+(1:p₂)
            ## copy the (k+1)'st square block of s.A₂₂ to a diagonal block of s.L₂₂,
            wL = copy!(view(L₂₂d,kk,kk),contiguous_view(s.A₂₂,k*p₂*p₂,(p₂,p₂)))
            Ac_mul_B!(λ₂,A_mul_B!(wL,λ₂)) # convert the block to λ₂A₂₂λ₂'
            for j in 1:p₂ # inflate diagonal and zero the strict upper triangle
                wL[j,j] += 1.
                for i in 1:(j-1)
                    wL[i,j] = 0.
                end
            end
            A_mul_B!(view(L₃₂,:,kk),λ₂)
        end
    end
    BLAS.syrk!('L','N',-1.,L₂₁,1.,L₂₂d)
    _, info = LAPACK.potrf!('L',L₂₂d)
    info == 0 ||  error("downdated Z₂'Z₂ is not positive definite")
    A_rdiv_Bc!(BLAS.gemm!('N','T',-1.,L₃₁,L₂₁,1.,L₃₂),s.L₂₂)
    L₃₃d = copy!(s.L₃₃.UL,s.A₃₃)
    BLAS.syrk!('L','N',-1.,L₃₁,1.,BLAS.syrk!('L','N',-1.,L₃₂,1.,L₃₃d))
    _, info = LAPACK.potrf!('L',L₃₃d)
    info == 0 ||  error("downdated X'X is not positive definite")
    s
end

## grad calculation
## need tr((LL')⁻¹*λ*Z'Z*(∂λ/∂θᵢ)) + tr((LL')⁻¹*(∂λ'/∂θᵢ)*Z'Z*λ)
## It may be worthwhile calculating and storing L⁻¹ for this.
## For parameters in λ₂ only the trailing rows of L⁻¹ are used.
