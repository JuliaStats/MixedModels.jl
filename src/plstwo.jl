type PLSTwo <: PLSSolver   # Solver for models with two crossed or nearly crossed terms
    Ad::Array{Float64,3}                # diagonal blocks
    Ab::Array{Float64,3}                # base blocks
    At::Symmetric{Float64}              # lower right block
    Ld::Array{Float64,3}                # diagonal blocks
    Lb::Array{Float64,3}                # base blocks
    Lt::Base.LinAlg.Cholesky{Float64}   # lower right triangle
    Zt::SparseMatrixCSC
    p::Int
    p2::Int
end

function PLSTwo(facs::Vector,Xst::Vector,Xt::Matrix)
    length(facs) == length(Xst) == 2 || throw(DimensionMismatch("PLSTwo"))
    L = size(Xt,2)
    all(L .== [size(x,2) for x in Xst]) && all(L .== map(length,facs)) || throw(DimensionMismatch(""))
    nl = [length(f.pool) for f in facs]
    pv = [size(x,1) for x in Xst]
    q = nl .* pv
                                        # Do this in lmm instead of checking here
    q[1] >= q[2] || error("reverse the order of the random effects terms")
    p = size(Xt,1)
    p2 = pv[2]
    Ad = zeros(pv[1],pv[1],nl[1])
    Ab = zeros(q[2]+p,pv[1],nl[1])
    finds = q[2] + (1:p)         # indices for the fixed-effects parts
    Abb = view(Ab,finds,:,:)
    At = zeros(q[2]+p,q[2]+p)
    BLAS.syrk!('L','N',1.0,Xt,0.0,view(At,finds,finds))
    r1 = facs[1].refs
    r2 = facs[2].refs
    xst1 = Xst[1]
    xst2 = Xst[2]
    for j in 1:L
        j1 = r1[j]
        inds2 = (r2[j]-1)*p2 + (1:p2)
        BLAS.syr!('L',1.0,view(xst1,:,j),view(Ad,:,:,j1))
        BLAS.syr!('L',1.0,view(xst2,:,j),view(At,inds2,inds2))
        BLAS.ger!(1.0,view(xst2,:,j),view(xst1,:,j),view(Ab,inds2,:,j1))
        BLAS.ger!(1.0,view(Xt,:,j),view(xst1,:,j),view(Abb,:,:,j1))
        BLAS.ger!(1.0,view(Xt,:,j),view(xst2,:,j),view(At,finds,inds2))
    end
    Lt = copy(At)
    for j in 1:q[2]          # inflate diagonal before taking Cholesky
        Lt[j,j] += 1.
    end
    PLSTwo(Ad,Ab,Symmetric(At,:L),similar(Ad),similar(Ab),cholfact!(Lt,:L),
           vcat(ztblk(xst1,r1),ztblk(xst2,r2)),p,p2)
end

function updateLdb!(s::Union(PLSOne,PLSTwo),λ::AbstractPDMatFactor)
    k = dim(λ)
    k < 0 || k == size(s.Ad,1) || throw(DimensionMixmatch(""))
    m,n,l = size(s.Ab)
    Lt = tril!(copy!(s.Lt.UL,s.At.S))
    Lb = copy!(s.Lb,s.Ab)
    if n == 1                           # shortcut for 1×1 λ
        isa(λ,PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ.s.λ
        lamsq = lam*lam
        Ld = map!(x -> sqrt(x*lamsq + 1.), s.Ld, s.Ad)
        scale!(reshape(Lb,(m,l)),lam ./ vec(Ld))
    else
        Ac_mul_B!(λ,reshape(copy!(s.Ld,s.Ad),(n,n*l)))
        for k in 1:l
            wL = A_mul_B!(view(s.Ld,:,:,k),λ)
            for j in 1:n
                wL[j,j] += 1.0
            end
            _,info = LAPACK.potrf!('L',wL)
            info == 0 || error("Cholesky failure at L diagonal block $k")
            Base.LinAlg.A_rdiv_Bc!(A_mul_B!(view(s.Lb,:,:,k),λ),Triangular(wL,:L,false))
        end
    end
    s
end

function update!(s::PLSTwo,λ::Vector)
    p2 = s.p2
    length(λ) == 2 && Base.LinAlg.chksquare(λ[2]) == p2 || throw(DimensionMismatch(""))
    updateLdb!(s,λ[1])
                                        # second level updates
    m,n,l = size(s.Lb)
    q2 = m - s.p
    rem(q2,p2) == 0 || throw(DimensionMismatch(""))
    Lt = Base.LinAlg.full!(s.Lt[:L])
    if p2 == 1
        lam = lfactor(λ[2])[1,1]
        scale!(lam,reshape(view(s.Lb,1:q2,:,:),(q2,l)))
        scale!(lam*lam,view(Lt[:L],1:q2,1:q2))
    else
        lam = λ[2]
        for k in 1:div(q2,p2)
            ii = (k - 1)*p2 + (1:p2)
            A_mul_B!(lam,reshape(view(s.Lb,ii,:,:),(p2,n*l)))
            A_mul_B!(lam,A_mul_Bc!(view(Lt,1:q2,1:q2),lam))
        end
    end
    for j in 1:q2
        Lt[j,j] += 1.
    end
    BLAS.syrk!('L','N',-1.0,reshape(s.Lb,(m,n*l)),1.0,Lt)
    _, info = LAPACK.potrf!('L',Lt)
    info == 0 ||  error("downdated X'X is not positive definite")
    s
end
