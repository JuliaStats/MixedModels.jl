using MixedModels
using Test

using MixedModels: dataset, likelihoodratiotest

include("modelcache.jl")

@testset "markdown" begin
    gm3 = fit(MixedModel, only(gfms[:verbagg]), dataset(:verbagg), Bernoulli())
    fm1 = last(models(:sleepstudy))
    fm0 = first(models(:sleepstudy))
    lrt = likelihoodratiotest(fm0, fm1)
    mime = MIME"text/markdown"()

    @test sprint(show, mime, fm0) == """
| |Est.|SE |z  |p  | σ_subj|
|:-|----:|--:|--:|--:|------:|
|(Intercept)|251.41|9.51|26.45|<1e-99|36.01|
|days|10.47|0.8|13.06|<1e-38||
|Residual|30.9|||||
"""

    @test sprint(show, mime, fm1) == """
| |Est.|SE |z  |p  | σ_subj|
|:-|----:|--:|--:|--:|------:|
|(Intercept)|251.41|6.63|37.91|<1e-99|23.78|
|days|10.47|1.5|6.97|<1e-11|5.72|
|Residual|25.59|||||
"""

    @test sprint(show, mime, gm3) == """
| |Est.|SE |z  |p  | σ_subj|σ_item|
|:-|----:|--:|--:|--:|------:|------:|
|(Intercept)|0.2|0.41|0.48|0.6294|1.34|0.5|
|anger|0.06|0.02|3.43|0.0006|||
|gender: M|0.32|0.19|1.68|0.0935|||
|btype: scold|-1.06|0.26|-4.12|<1e-04|||
|btype: shout|-2.1|0.26|-8.14|<1e-15|||
|situ: self|-1.05|0.21|-5.02|<1e-06|||
"""

    @test sprint(show, mime, lrt) == """
||model-dof|deviance|χ²|χ²-dof|P(>χ²)|
|:-|-:|-:|-:|-:|:-|
|reaction ~ 1 + days + (1 \\| subj)|4|1794| | | |
|reaction ~ 1 + days + (1 + days \\| subj)|6|1752|42|2|<1e-09|
"""

    @test sprint(show, mime, BlockDescription(gm3)) == """
|rows|subj|item|fixed|
|:--|:--:|:--:|:--:|
|316|Diagonal|||
|24|Dense|Diag/Dense||
|6|Dense|Dense|Dense|
"""

    @test sprint(show, mime, fm1.optsum) == """
| | |
|-|-|
|**Initialization**| |
|Initial parameter vector|[1.0, 0.0, 1.0]|
|Initial objective value|1784.642296192471|
|**Optimizer settings**| |
|Optimizer (from NLopt)|LN_BOBYQA|
|`Lower bounds`|[0.0, -Inf, 0.0]|
|`ftol_rel`|1.0e-12|
|`ftol_abs`|1.0e-8|
|`xtol_rel`|0.0|
|`xtol_abs`|[1.0e-10, 1.0e-10, 1.0e-10]|
|`initial_step`|[0.75, 1.0, 0.75]|
|`maxfeval`|-1|
|**Result**| |
|Function evaluations|57|
|Final parameter vector|[0.9292, 0.0182, 0.2226]|
|Final objective value|1751.9393|
|Return code|`FTOL_REACHED`|
"""


    @test sprint(show, mime, VarCorr(fm1)) == """
|   |Column|Variance|Std.Dev.|Corr.|
|:--|:-----|-------:|-------:|----:|
|subj|(Intercept)|565.51069|23.78047| |
| |days|32.68212|5.71683| +0.08|
|Residual| |654.94145|25.59182|
"""

    @test sprint(show, mime, VarCorr(gm3)) == """
|   |Column|Variance|Std.Dev.|
|:--|:-----|-------:|-------:|
|subj|(Intercept)|1.794973|1.339766|
|item|(Intercept)|0.245327|0.495305|
"""

end
