## Solver for models with two or more grouping factors forming a nested sequence

## Let nf be the number of nested grouping factors in the model, nl be
## the (length nf) vector of the number of levels in the grouping
## factors and pv be the number of rows in the transposed model
## matrices.  nl must be non-increasing.  Amats is a (length
## nf+1) vector of matrices. The i'th matrix in Amats has [pv,p][i]
## rows and cumsum([nl.*pv,p])[i] columns. Lmats is a (length nf+1)
## vector of matrices similar to Amats, containing the corresponding
## parts of the lower Cholesky factor. (There is no fill-in when
## factoring the system matrix from nested grouping factors.)
## Amats[1] is logically divided into nl[1] matrices of size
## pv[1]×pv[1]. Amats[2] is logically divided into nl[1] matrices of
## size pv[2]×pv[1] followed by nl[2] matrices of size pv[2]×pv[2],
## etc.  The vector inds[i] consists of i-1 Vector{UnitRange{Int}}
## objects, each of length nl[i].  The ranges are the columns in
## Amats[i] corresponding to facs[j] by facs[i] cross-terms (j < i)
## for each level of facs[i].

type PLSNested <: PLSSolver
    Amats::Vector                       # arrays of cross-products
    Lmats::Vector                       # storage for factors
    lastij::Vector  # last level of factor i nested in a level of factor j, i < j
    nl::Vector{Int} # number of levels of factor i (fixed effects have 1 level)
    offsets::Vector{Int}     # offsets of column range for factor i
    pv::Vector{Int}          # number of rows in each element of Lmats
    gtmp::Vector{Matrix{Float64}}       # scratch arrays for grad calculation
end

## On entry facs is sorted by decreasing levels and nesting of facs has been checked.
function PLSNested(facs::Vector,Xst::Vector,Xt::Matrix)
                                        # check dimensions
    (nf = length(facs)) == length(Xst) || throw(DimensionMismatch(""))
    p,n = size(Xt)
    nl = [[length(f.pool) for f in facs],1]
    issorted(nl;rev=true) || error("facs must be sorted by decreasing numbers of levels")
    pv = [[size(xst,1)::Int for xst in Xst],p]
                                        # check that the facs refs are increasing
    refs = [f.refs for f in facs]
    push!(refs,ones(Uint8,n))
    issorted(collect(zip(refs[nf:-1:1]...))) ||
        error("facs refs must be in increasing lexicographic order")
                                        # determine offsets for each factor
    nfp1 = nf + 1
    Amats = Array(Matrix{Float64},nfp1)
    offsets = Array(Int,nfp1)
    ncols = 0
    for j in 1:nfp1
        offsets[j] = ncols
        ncols += nl[j] * pv[j]
        Amats[j] = zeros(pv[j],ncols)
    end
    ## may need to generalize this approach.  See mlmRev's egsingle for an example
    ## where this approach fails.
    ## Determine a permutation from the observed refs to the desired refs as in
    ## unique(egs[:ChildId].refs)
    ## replace the refs for ChildId by invperm(unique(egs[:ChildId].refs))[egs[:ChildId].refs]
                                        # determine the change points for each grouping factor
    chgpts = [Array(Int,(nl[j],)) for j in 1:nfp1] 
    for i in 1:n
        for j in 1:nfp1
            chgpts[j][refs[j][i]] = i
        end
    end

    lastij = Array(Array{Array{Int,1},1},nfp1)
    for j in 1:nfp1
        indsj = [[0] for _ in 1:(j-1)]
        for cp in chgpts[j]
            for k in 1:(j-1)
                rng = searchsorted(chgpts[k],cp)
                length(rng) == 1 || error("facs are not nested")
                push!(indsj[k],first(rng))
            end
        end
        for k in 1:(j-1)                # convert to columns
            indsjk = indsj[k]
            for i in 1:length(indsjk)
                indsjk[i] = indsjk[i] * pv[k] + offsets[k]
            end
        end
        lastij[j] = indsj
    end
                                        # populate Amats
    vecs = Array(DenseVector{Float64},nfp1) # holds the j'th column of Xst[i], i=1,...,nf
    for j in 1:n
        for i in 1:nf
            vecs[i] = view(Xst[i],:,j)
        end
        vecs[nfp1] = view(Xt,:,j)
        for i in 1:nfp1
            colrng = offsets[i]+(refs[i][j]-1)*pv[i]+(1:pv[i])
            for k in i:nfp1
                BLAS.ger!(1.,vecs[k],vecs[i],view(Amats[k],:,colrng))
            end
        end
    end
    pf = view(pv,1:nf)
    pmax = maximum(pf)
    PLSNested(Amats,[zeros(a) for a in Amats],lastij,nl,offsets,pv,[zeros(p,pmax) for p in pf])
end

## argument uβ contains vcat(λ'Z'y,X'y) on entry
function Base.A_ldiv_B!(s::PLSNested,uβ::Vector{Float64})
    length(uβ) == size(s.Lmats[end],2) || throw(DimensionMismatch(""))
    pv = s.pv
    nl = s.nl
    offsets = s.offsets
    Lmats = s.Lmats
    nm = length(Lmats)
                                        # forward solve
    for i in 1:nm
        lst = s.lastij[i]
        Lmi = Lmats[i]
        colrng = offsets[i] + (1:pv[i])
        for k in 1:nl[i]
            vik = view(uβ,colrng)
            for j in 1:(i-1)
                kjrng = (lst[j][k]+1):lst[j][k+1]
                BLAS.gemv!('N',-1.0,view(Lmi,:,kjrng),view(uβ,kjrng),1.0,vik)
            end
            BLAS.trsv!('L','N','N',view(Lmi,:,colrng),vik)
            colrng += pv[i]
        end        
    end
                                        # backsolve
    for i in nm:-1:1
        lst = s.lastij[i]
        pvi = pv[i]
        Lmi = s.Lmats[i]
        colrng = offsets[i] + (1:pvi)
        for k in 1:nl[i]
            vik = view(uβ,colrng)
            BLAS.trsv!('L','T','N',view(Lmi,:,colrng),vik)
            for j in 1:(i-1)
                kjrng = (lst[j][k]+1):lst[j][k+1]
                BLAS.gemv!('T',-1.0,view(Lmi,:,kjrng),vik,1.0,view(uβ,kjrng))
            end
            colrng += pvi
        end
    end
end

function Base.cholfact(s::PLSNested,RX::Bool=true)
    Lmats = s.Lmats
    if RX
        nm = length(Lmats)
        Lm = Lmats[nm]
        return cholesky(Lm[:,(s.offsets[nm]+1:size(Lm,2))],:L)
    end
    error("code not yet written")
end

function Base.full(s::PLSNested)
    offsets = s.offsets
    ntot = offsets[end]
    A = zeros(ntot,ntot)
    L = zeros(ntot,ntot)
    for i in 1:(length(offsets) - 1)    # random-effects terms only
        Ai = s.Amats[i]
        Li = s.Lmats[i]
        oo = offsets[i]
        pvi = s.pv[i]
        lij = s.lastij[i]
        for j in 1:s.nl[i]
            rb = oo + (j-1)*pvi         # row base
            for ℓ in 1:pvi              # fill in the diagonal
                for k in 1:pvi
                    A[rb+k,rb+ℓ] = Ai[k,rb+ℓ]
                    k ≤ ℓ && (L[rb+k,rb+ℓ] = Li[k,rb+ℓ])
                end
                for ll in lij
                    for jj in (ll[j]+1):ll[j+1]
                        for k in 1:pvi
                            A[rb+k,jj] = Ai[k,jj]
                            L[rb+k,jj] = Li[k,jj]
                        end
                    end
                end
            end
        end
    end
    (Symmetric(A,:L),cholesky(L,:L))
end

## return a view of the diagonal block of L at level j for index k of level i
function dblk(i,j,k,s)
    j ≥ i || error("j = $j must be ≥ i = $i")
    if j == i
        pvi = s.pv[i]
        return view(s.Lmats[i],:,s.offsets[i] + (k-1)*pvi + (1:pvi))
    end
    pvj = s.pv[j]
    view(s.Lmats[j],:,s.offsets[j]+(searchsortedfirst(s.lastij[j][i],k)-2)*pvj+(1:pvj))
end

## Evaluate the contribution to the gradient from the logdet term
function grad!(res::Vector{Matrix{Float64}},s::PLSNested,λ::Vector)
    nf = length(s.Amats)-1
    for i in 1:nf                       # factor i
        p = s.pv[i]
        indsi = s.offsets[i] + (1:p)
        for k in 1:s.nl[i]              # level k of factor i
            for j in i:nf               # store and downdate blocks of Λ'Z'Z
                gt = view(s.gtmp[j],:,1:p)
                Ac_mul_B!(λ[j],copy!(gt,view(s.Amats[j],:,indsi)))
                for ii in 1:(i-1)
                    ll = s.lastij[j][ii]
                    pii = s.pv[ii]
                    gtt = view(s.gtmp[ii],:,1:p)
                    for kk in (ll[k]):pii:(ll[k+1]-1)
                        BLAS.gemm!('N','N',-1.,view(s.Lmats[j],:,kk+(1:pii)),
                                   A_ldiv_B!(Triangular(view(s.Lmats[ii],:,kk+(1:pii)),:L,false),
                                             transpose!(gtt,view(s.Amats[j],:,kk+(1:pii)))),
                                   1.,gt)
                    end
                end
            end
            for j in i:nf               # forward solve
                vv = view(s.gtmp[j],:,1:p)
                A_ldiv_B!(Triangular(dblk(i,j,k,s),:L,false),vv)
                for jj in (j+1):nf
                    BLAS.gemm!('N','N',-1.0,view(s.Lmats[jj],:,indsi),vv,1.0,view(s.gtmp[jj],:,1:p))
                end
            end
            @show i,k
            @show s.gtmp
            for j in nf:-1:i            # backsolve
                vv = view(s.gtmp[j],:,1:p)
                Ac_ldiv_B!(Triangular(dblk(i,j,k,s),:L,false),vv)
                for jj in i:(j-1)
                    BLAS.gemm!('T','N',-1.0,view(s.Lmats[j],:,indsi),vv,1.0,view(s.gtmp[jj],:,1:p))
                end
            end
            @show s.gtmp
            indsi += p
        end
    end
end

               
## Logarithm of the determinant of the matrix represented by RX or L
function Base.logdet(s::PLSNested,RX=true)
    nm = length(s.Lmats)
    sm = 0.
    if RX
        Lm = view(s.Lmats[nm],:,(s.offsets[nm]+1):size(s.Lmats[nm],2))
        for i in 1:size(Lm,1)
            sm += log(Lm[i,i])
        end
    else
        for i in 1:(nm - 1)
            pvi = s.pv[i]
            Lmi = s.Lmats[i]
            colrng = s.offsets[i] + (1:pvi)
            for k in 1:s.nl[i]
                lmik = view(Lmi,:,colrng)
                for j in 1:pvi
                    sm += log(lmik[j,j])
                end
                colrng += pvi
            end
        end
    end
    2.sm
end

function Base.sparse(s::PLSNested)
    offsets = s.offsets
    ntot = offsets[end]
    nf = length(offsets) - 1
    nnzL = mapreduce(length, +, s.Lmats[1:nf])
    LI = Int32[]; LJ = Int32[]; LV = Float64[]
    sizehint(LI,nnzL); sizehint(LJ,nnzL); sizehint(LV,nnzL)
    for i in 1:nf    # random-effects terms only
        Li = s.Lmats[i]
        oo = offsets[i]
        pvi = s.pv[i]
        lij = s.lastij[i]
        for j in 1:s.nl[i]
            rb = oo + (j-1)*pvi         # row base
            for ℓ in 1:pvi              # fill in the diagonal
                for k in 1:ℓ
                    push!(LI,rb+k)
                    push!(LJ,rb+ℓ)
                    push!(LV,Li[k,rb+ℓ])
                end
                for ll in lij
                    for jj in (ll[j]+1):ll[j+1]
                        for k in 1:pvi
                            push!(LI,rb+k)
                            push!(LJ,jj)
                            push!(LV,Li[k,jj])
                        end
                    end
                end
            end
        end
    end
    Triangular(sparse(LI,LJ,LV,ntot,ntot),:L,false)
end
    

##  update!(s,λ)->s : update Lmats, the Cholesky factor
function update!(s::PLSNested,λ::Vector)
    length(λ) == (nf = (nfp1 = length(s.Amats)) - 1) || throw(DimensionMismatch(""))
    offsets = s.offsets
    nl = s.nl
    pv = s.pv
    for i in 1:nfp1                     # copy Amats into Lmats
        copy!(s.Lmats[i],s.Amats[i])
    end
    for i in 1:nf                       # form Λ'Z'ZΛ + I in Lmats
        λi = λ[i]
        Lmi = s.Lmats[i]
        Ac_mul_B!(λi,Lmi)               # multiply by λᵢ′ on left
        colrng = offsets[i] + (1:pv[i]) # incremented by pv[i] for each level of i'th factor
        for k in 1:s.nl[i]
            for j in i:nfp1
                A_mul_B!(view(s.Lmats[j],:,colrng),λi) # multiply by λⱼ on the right
            end
            dbk = view(Lmi,:,colrng)    # k'th diagonal block of Lⱼⱼ
            for j in 1:pv[i]            # inflate the diagonal of dbk
                dbk[j,j] += 1.
                for ii in 1:(j-1)       # zero strict upper triangle (cosmetic)
                    dbk[ii,j] = 0.
                end
            end
            colrng += pv[i]
        end
    end
    for i in 1:nfp1                     # create the Cholesky factor
        Lmi = s.Lmats[i]
        colrng = offsets[i] + (1:pv[i]) # incremented by pv[i] for each level of i'th factor
        for k in 1:s.nl[i]
            dbk = view(Lmi,:,colrng)    # diagonal block
            for ii in s.lastij[i]       # downdate by cross-products of blocks to the left
                krng = (ii[k]+1):ii[k+1]
                Lmik = view(Lmi,:,krng)
                BLAS.syrk!('L','N',-1.0,Lmik,1.0,dbk)
                for j in (i+1):nfp1
                    BLAS.gemm!('N','T',-1.0,view(s.Lmats[j],:,krng),Lmik,
                               1.0,view(s.Lmats[j],:,colrng))
                end
            end 
            _,info = LAPACK.potrf!('L',dbk)
            info==0 || error("Downdated diagonal block $k for term $i is not positive definite")
            for j in (i+1):nfp1         # evaluate the remaining blocks in colrng
                A_rdiv_Bc!(view(s.Lmats[j],:,colrng),Triangular(dbk,:L,false))
            end
            colrng += pv[i]
        end
    end
    s
end
