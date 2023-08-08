using MixedModels
using Suppressor
using Test

using MixedModels: dataset, likelihoodratiotest
using MixedModels: pirls!, setβθ!, setθ!, updateL!

include("modelcache.jl")

# explicitly setting theta for these to so that we can do exact textual comparisons
βθ = [0.1955554704948119,  0.05755412761885973, 0.3207843518569843, -1.0582595252774376,
     -2.1047524824609853, -1.0549789653925743,  1.339766125847893,  0.4953047709862237]
gm3 = GeneralizedLinearMixedModel(only(gfms[:verbagg]), dataset(:verbagg), Bernoulli())
pirls!(setβθ!(gm3, βθ))

fm0θ = [1.1656121258575225]
fm0 = updateL!(setθ!(first(models(:sleepstudy)), fm0θ))
fm0.optsum.feval = 1

fm1θ = [0.9292213288149662, 0.018168393450877257, 0.22264486671069741]
fm1 = updateL!(setθ!(last(models(:sleepstudy)), fm1θ))
fm1.optsum.feval = 1

fmreθ = [0.32352483854887326, 0.4715395478019364, 0.0,
         0.43705610601403755, 0.016565641868150047, 0.17732248078617097]
# this is a junk model, but it stresses parts of the display code
fmre = LinearMixedModel(@formula(rt_trunc ~ 1+(0+spkr|subj)+(1+load|item)), MixedModels.dataset(:kb07))
updateL!(setθ!(fmre, fmreθ))
fmre.optsum.feval = 1

lrt = likelihoodratiotest(fm0, fm1)

@testset "markdown" begin
    mime = MIME"text/markdown"()
    gm3.optsum.feval = -1
    @test_logs (:warn, "Model has not been fit: results will be nonsense") sprint(show, mime, gm3)
    gm3.optsum.feval = 1
    @testset "lmm" begin
        @test sprint(show, mime, fm0) == """
|             |     Est. |     SE |     z |      p |  σ_subj |
|:----------- | --------:| ------:| -----:| ------:| -------:|
| (Intercept) | 251.4051 | 9.5062 | 26.45 | <1e-99 | 36.0121 |
| days        |  10.4673 | 0.8017 | 13.06 | <1e-38 |         |
| Residual    |  30.8954 |        |       |        |         |
"""
        @test sprint(show, mime, fm1) == """
|             |     Est. |     SE |     z |      p |  σ_subj |
|:----------- | --------:| ------:| -----:| ------:| -------:|
| (Intercept) | 251.4051 | 6.6323 | 37.91 | <1e-99 | 23.7805 |
| days        |  10.4673 | 1.5022 |  6.97 | <1e-11 |  5.7168 |
| Residual    |  25.5918 |        |       |        |         |
"""
    end

    @testset "re without fe" begin
        @test sprint(show, mime, fmre) == """
|             |      Est. |      SE |     z |      p |   σ_subj |   σ_item |
|:----------- | ---------:| -------:| -----:| ------:| --------:| --------:|
| (Intercept) | 2092.3713 | 76.9426 | 27.19 | <1e-99 |          | 349.7858 |
| spkr: old   |           |         |       |        | 377.3837 |          |
| spkr: new   |           |         |       |        | 258.9242 |          |
| load: yes   |           |         |       |        |          | 142.5331 |
| Residual    |  800.3224 |         |       |        |          |          |
"""
    end

    @testset "glmm" begin
        @test sprint(show, mime, gm3) in ("""
|              |    Est. |     SE |     z |      p | σ_subj | σ_item |
|:------------ | -------:| ------:| -----:| ------:| ------:| ------:|
| (Intercept)  |  0.1956 | 0.4052 |  0.48 | 0.6294 | 1.3398 | 0.4953 |
| anger        |  0.0576 | 0.0168 |  3.43 | 0.0006 |        |        |
| gender: M    |  0.3208 | 0.1913 |  1.68 | 0.0935 |        |        |
| btype: scold | -1.0583 | 0.2568 | -4.12 | <1e-04 |        |        |
| btype: shout | -2.1048 | 0.2585 | -8.14 | <1e-15 |        |        |
| situ: self   | -1.0550 | 0.2103 | -5.02 | <1e-06 |        |        |
""","""
|              |    Est. |     SE |     z |      p | σ_subj | σ_item |
|:------------ | -------:| ------:| -----:| ------:| ------:| ------:|
| (Intercept)  |  0.1956 | 0.4052 |  0.48 | 0.6294 | 1.3398 | 0.4953 |
| anger        |  0.0576 | 0.0168 |  3.43 | 0.0006 |        |        |
| gender: M    |  0.3208 | 0.1913 |  1.68 | 0.0935 |        |        |
| btype: scold | -1.0583 | 0.2568 | -4.12 |  <1e-4 |        |        |
| btype: shout | -2.1048 | 0.2585 | -8.14 | <1e-15 |        |        |
| situ: self   | -1.0550 | 0.2103 | -5.02 |  <1e-6 |        |        |
""")
    end

    @testset "lrt" begin

        @test sprint(show, mime, lrt) in ("""
|                                          | model-dof | deviance |  χ² | χ²-dof | P(>χ²) |
|:---------------------------------------- | ---------:| --------:| ---:| ------:|:------ |
| reaction ~ 1 + days + (1 \\| subj)        |         4 |     1794 |     |        |        |
| reaction ~ 1 + days + (1 + days \\| subj) |         6 |     1752 |  42 |      2 | <1e-09 |
""","""
|                                          | model-dof | deviance |  χ² | χ²-dof | P(>χ²) |
|:---------------------------------------- | ---------:| --------:| ---:| ------:|:------ |
| reaction ~ 1 + days + (1 \\| subj)        |         4 |     1794 |     |        |        |
| reaction ~ 1 + days + (1 + days \\| subj) |         6 |     1752 |  42 |      2 | <1e-9  |
""")
    end


    @testset "blockdescription" begin

        @test sprint(show, mime, BlockDescription(gm3)) == """
| rows | subj     | item       | fixed |
|:---- |:-------- |:---------- |:----- |
| 316  | Diagonal |            |       |
| 24   | Dense    | Diag/Dense |       |
| 7    | Dense    | Dense      | Dense |
"""
    end


    @testset "optsum" begin
        fm1.optsum.feval = 1
        fm1.optsum.initial_step = [0.75, 1.0, 0.75]
        fm1.optsum.finitial = 1784.642296192471
        fm1.optsum.final = [0.9292, 0.0182, 0.2226]
        fm1.optsum.fmin =1751.9393444647023
        out =  sprint(show, mime, fm1.optsum)
        @test startswith(out,"""
|                          |                             |
|:------------------------ |:--------------------------- |
| **Initialization**       |                             |
| Initial parameter vector | [1.0, 0.0, 1.0]             |
| Initial objective value  | 1784.642296192471           |
| **Optimizer settings**   |                             |
| Optimizer (from NLopt)   | `LN_BOBYQA`                 |
| Lower bounds             | [0.0, -Inf, 0.0]            |""")
    end


    @testset "varcorr" begin

        @test sprint(show, mime, VarCorr(fm1)) == """
|          | Column      |  Variance |  Std.Dev | Corr. |
|:-------- |:----------- | ---------:| --------:| -----:|
| subj     | (Intercept) | 565.51069 | 23.78047 |       |
|          | days        |  32.68212 |  5.71683 | +0.08 |
| Residual |             | 654.94145 | 25.59182 |       |
"""

        @test sprint(show, mime, VarCorr(gm3)) == """
|      | Column      |  Variance |  Std.Dev |
|:---- |:----------- | ---------:| --------:|
| subj | (Intercept) |  1.794973 | 1.339766 |
| item | (Intercept) |  0.245327 | 0.495305 |
"""
    end
end

@testset "html" begin
    # this is minimal since we're mostly testing that dispatch works
    # the stdlib actually handles most of the conversion
    @test sprint(show, MIME("text/html"), BlockDescription(gm3)) == """
<table><tr><th align="left">rows</th><th align="left">subj</th><th align="left">item</th><th align="left">fixed</th></tr><tr><td align="left">316</td><td align="left">Diagonal</td><td align="left"></td><td align="left"></td></tr><tr><td align="left">24</td><td align="left">Dense</td><td align="left">Diag/Dense</td><td align="left"></td></tr><tr><td align="left">7</td><td align="left">Dense</td><td align="left">Dense</td><td align="left">Dense</td></tr></table>
"""
    optsum = sprint(show, MIME("text/html"), fm0.optsum)

    @test occursin("<b>Initialization</b>", optsum)
    @test occursin("<code>LN_BOBYQA</code>", optsum)
end

@testset "latex" begin
    # this is minimal since we're mostly testing that dispatch works
    # the stdlib actually handles most of the conversion

    b = BlockDescription(gm3)

    @test sprint(show, MIME("text/latex"), b) == """
\\begin{tabular}
{l | l | l | l}
rows & subj & item & fixed \\\\
\\hline
316 & Diagonal &  &  \\\\
24 & Dense & Diag/Dense &  \\\\
7 & Dense & Dense & Dense \\\\
\\end{tabular}
"""

    @test sprint(show, MIME("text/xelatex"), b) == sprint(show, MIME("text/latex"), b)

    @test sprint(show, MIME("text/xelatex"), gm3) != sprint(show, MIME("text/latex"), gm3)

    @test startswith(sprint(show, MIME("text/latex"), gm3),"""
\\begin{tabular}
{l | r | r | r | r | r | r}
 & Est. & SE & z & p & \$\\sigma_\\text{subj}\$ & \$\\sigma_\\text{item}\$ \\\\""")

    # not doing the full comparison here because there's a zero-padded exponent
    # that will render differently on different platforms
    @test startswith(sprint(show, MIME("text/latex"), lrt),
                     "\\begin{tabular}\n{l | r | r | r | r | l}\n & model-dof & deviance & \$\\chi^2\$ & \$\\chi^2\$-dof & P(>\$\\chi^2\$) \\\\")


    optsum = sprint(show, MIME("text/latex"), fm0.optsum)

    @test occursin(raw"\textbf{Initialization}", optsum)
    @test occursin(raw"\texttt{LN\_BOBYQA}", optsum)
end

# return these models to their fitted state for the cache
refit!(fm1; progress=false)
refit!(fm0; progress=false)
