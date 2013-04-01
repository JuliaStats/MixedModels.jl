## Some of the examples from the lme4 package for R

## Right now I don't have a formula/data interface working so these examples look awkward

using DataFrames, RDatasets, Distributions, MixedModels

ds = data("lme4", "Dyestuff")
fm1 = fit(LMMsimple(int32(ds[:Batch].refs)'', ones(size(ds,1),1), ds[:Yield].data), true);
show(fm1)

psts = data("lme4", "Pastes")
fm2 = fit(LMMsimple(int32(hcat(psts[:sample].refs,psts[:batch].refs)), ones(size(psts,1),1), psts[:strength].data), true);
show(fm2)

pen = data("lme4", "Penicillin")
fm3 = fit(LMMsimple(int32(hcat(pen[:plate].refs,pen[:sample].refs)), ones(size(pen,1),1), pen[:diameter].data), true);
show(fm3)

chem = data("mlmRev", "Chem97")
fm4 = fit(LMMsimple(int32(hcat(chem[:school].refs,chem[:lea].refs)), ones(size(chem,1),1), chem[:score].data), true);
show(fm4)

fm5 = fit(LMMsimple(int32(hcat(chem[:school].refs,chem[:lea].refs)), hcat(ones(size(chem,1)),chem[:gcsecnt].data), chem[:score].data), true);
show(fm5)

fm6 = fit(reml(LMMsimple(int32(hcat(chem[:school].refs,chem[:lea].refs)), hcat(ones(size(chem,1)),chem[:gcsecnt].data), chem[:score].data)), true);
show(fm6)

@elapsed fm6 = fit(reml(LMMsimple(int32(hcat(chem[:school].refs,chem[:lea].refs)), hcat(ones(size(chem,1)),chem[:gcsecnt].data), chem[:score].data)))


