type Pedigree
    sire::Vector{Int}
    dam::Vector{Int}
    function Pedigree(sire::Vector{Int},dam::Vector{Int})
        (n = length(sire)) == length(dam) || throw(DimensionMismatch(""))
        ordered = true
        for i in 1:n
            0 <= sire[i] <= n && 0 <= dam[i] <= n || error("sire and dam must be in 0:n")
            sire[i] < i && dam[i] < i || (ordered = false)
        end
        @show ordered
        new(sire,dam)
    end
end

function inbreeding(p::Pedigree)
    sire = p.sire
    dam = p.dam
    n = length(sire)
    F = zeros(n+1)          # inbreeding coefficients
    L = zeros(n+1)
    B = zeros(n)
    Anc = zeros(Int,n+1)    # Ancestor
    LAP = zeros(Int,n)      # longest ancestroral path
    F[1] = -1.              # initialize F and LAP for unknown parents
    LAP[1] = -1
                            # evaluate LAP and its maximum
    len = 0
    for i in 1:n
        S = sire[i] == 0 ? -1 : LAP[sire[i]]
        D = dam[i] == 0 ? -1 : LAP[dam[i]]
        LAP[i] = max(S,D) + 1
        len = max(len, LAP[i])
    end
    SI = zeros(Int,n+1)
    MI = zeros(Int,n+1)
    for i in 1:n
        S = sire[i]
        D = dam[i]
        FS = S == 0 ? -1. : F[S]
        FD = D == 0 ? -1. : F[D]
        B[i] = 0.5 - 0.25 * (FS + FD)
                                        # adjust start and minor
        for j in 1:LAP[i]
            SI[j] += 1
            MI[j] += 1
        end
        if S == 0 && D == 0             # both parents unknown
            F[i] = L[i] = 0.
            continue
        end
        if i > 1 && S == sire[i-1] && D == dam[i-1] # full sib with last animal
            F[i] = F[i-1]
            L[i] = L[i-1]
            continue
        end
        F[i] = -1.
        L[i] = -1.
        t = LAP[i]
        Anc[MI[t]] = i
        MI[t] += 1
        while t ≥ 0
            j = Anc[MI[t] -= 1]         # next ancestor
            S = sire[j]                 # parents of the ancestor
            D = dam[j]
            if S > 0
                if (L[S] != 0)
                    Anc[MI[LAP[S]]] = S # add sire to Anc
                    MI[LAP[S]] += 1
                end
                L[S] += 0.5 * L[j]
            end
            if D > 0
                if (L[D] != 0)
                    Anc[MI[LAP[D]]] = D # add sire to Anc
                    MI[LAP[D]] += 1
                end
                L[D] += 0.5 * L[j]
            end
            F[i] += abs2(L[j]) * B[j]
            L[j] = 0.                   # clear L for evaluation of next animal
            @show i,j,t
            @show MI[t]
            @show SI[t]
            if MI[t] == SI[t]           # move to the next LAP group
                t -= 1
            end
       end     
    end
    F .+ 1.
end

function incr(LAP::Vector{Int},sire::Vector{Int},dam::Vector{Int},k::Int)
    sl = (S = sire[k]) == 0 ? -1 : (LAP[S] >= 0 ? LAP[S] : incr(LAP,sire,dam,S))
    dl = (D = dam[k]) == 0 ? -1 : (LAP[D] >= 0 ? LAP[D] : incr(LAP,sire,dam,D))
    @show k, S, sl, D, dl
    LAP[k] = max(sl,dl) + 1
end

function editped(sire::Vector{Int},dam::Vector{Int})
    (n = length(sire)) == length(dam) || throw(DimensionMismatch(""))
    LAP = fill(-1,n)                    # longest ancestor path
    for i in 1:n
        S = sire[i]
        D = dam[i]
        0 <= S <= n && 0 <= D <= n || error("all sire and dam values must be in [0,n]")
        if S == 0 && D == 0
            LAP[i] = 0
        end
    end
    for i in 1:n
        @show i,LAP[i]
        LAP[i] == -1 && incr(LAP,sire,dam,i)
        @show i,LAP[i]
    end
    LAP
end

function orderped(sire::Vector{Int},dam::Vector{Int})
    (n = length(sire)) == length(dam) || throw(DimensionMismatch(""))
    for i in 1:n
        0 ≤ sire[i] ≤ n && 0 ≤ dam[i] ≤ n || error("sire and dam must be in 0:n")
    end
    ord = Int[]
    pop = IntSet(1:n)
    ss = delete!(IntSet(sire),0)
    dd = delete!(IntSet(dam),0)
    parents = union(ss,dd)
    while length(parents) > 0
        append!(ord,collect(setdiff(pop,parents)))
        pp = collect(parents)
        ss = delete!(IntSet(sire[pp]),0)
        dd = delete!(IntSet(dam[pp]),0)
        pop = parents
        parents = union(ss,dd)
    end
    append!(ord,collect(pop))
    reverse(ord)
end
