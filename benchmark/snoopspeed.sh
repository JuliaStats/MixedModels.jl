#! /bin/bash

# See #608
# julia> using SnoopCompile, MixedModels
# julia> ttot, pcs = SnoopCompile.parcel(snoop)
# julia> snoop = @snoopi_deep  begin
#                    MixedModels.fit(MixedModel, @formula(reaction ~ 1 + days + (1+days|subj)), MixedModels.dataset(:sleepstudy))
#                    MixedModels.fit(MixedModel,@formula(use ~ 1+age+abs2(age)+urban+livch+(1|urban&dist)), MixedModels.dataset(:contra), Bernoulli())
#                end
# julia> SnoopCompile.write(joinpath(@__DIR__, "src"), pcs;has_bodyfunction=true)
#
# I then had to edit precompile_MixedModels.jl and replace a few type parameters that were still there
# (s/T<:Core.AbstractFloat/Float64/g and s/D<:Distributions.Distribution/Distributions.Bernoulli/g).
#
# Because of the way Tables.jl and StatsModels use NamedTuples, I also had to
# remove lines referencing specific variables in these models.

for i in {1..3}; do
    echo "Run $i"
    julia --project=. -e'using MixedModels; @time fit(MixedModel, @formula(rt_trunc ~ 1+spkr*prec*load+(1|subj)+(1+prec|item)), MixedModels.dataset(:kb07))'
    julia --project=. -e'using MixedModels; @time fit(MixedModel, @formula(r2 ~ 1+anger+gender+btype+situ+(1|subj)+(1|item)), MixedModels.dataset(:verbagg), Binomial())'
done
