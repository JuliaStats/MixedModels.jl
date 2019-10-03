import RCall: rcopy, RClass, rcopytype, S4Sxp

# from R
# note that weights are not extracted
function rcopy(::Type{LinearMixedModel}, s::Ptr{S4Sxp})
    f = rcopy(s[:call][:formula])
    data = rcopy(s[:frame])
    θ = rcopy(s[:theta])
    reml = rcopy(s[:devcomp][:dims][:REML]) ≠ 0

    m = LinearMixedModel(f,data)
    m.optsum.REML = reml
    m.optsum.feval = rcopy(s[:optinfo][:feval])
    updateL!(setθ!(m, θ))
end

rcopytype(::Type{RClass{:lmerMod}}, s::Ptr{S4Sxp}) = LinearMixedModel

#     jreml = ifelse(REML, "true", "false")
#
#     julia_assign("jlmerdat",data)
#     jcall <- sprintf("jm = fit!(LinearMixedModel(@formula(%s),jlmerdat),REML=%s);",jf,jreml)
#
#     jout <- julia_command(jcall)
#
#     joptimizerOutput <- list(par=julia_eval("jm.optsum.final"),
#                          fval=julia_eval("jm.optsum.fmin"),
#                          feval=julia_eval("jm.optsum.feval"),
#                          # MixedModels.jl doesn't yet implement a lot of the
#                          # post-fit convergence checks that lme4 does but a very
#                          # crude one is provided by checking whether we reached
#                          # iteration stop. Julia has a few good packages for
#                          # Automatic Differentiation, maybe it's worthwhile to
#                          # use those for the gradient and Hessian checks to
#                          # really speed things up?
#                          conv=julia_eval("jm.optsum.maxfeval") == julia_eval("jm.optsum.feval"),
#                          message=julia_eval("jm.optsum.returnvalue"),
#                          optimizer=julia_eval("jm.optsum.optimizer"))
#
#
#     # we could extract this from the julia object, but the parsing is quite fast
#     # in lme4 and then we don't need to worry about converting formats and
#     # datatypes
#
#     parsedFormula <- lFormula(formula=formula,
#                               data=data,
#                               REML=REML)
#     # this bit should probably be reworked to extract the julia fields
#     devianceFunction <- do.call(mkLmerDevfun, parsedFormula)
#     optimizerOutput <- optimizeLmer(devianceFunction,start=joptimizerOutput$par,
#                                     control=list(maxeval=1))
#     optimizerOutput$feval <- joptimizerOutput$feval
#     optimizerOutput$message <- joptimizerOutput$message
#     optimizerOutput$optimizer <- joptimizerOutput$optimizer
#
#     rho <- environment(devianceFunction)
#     # especially rho$resp seems relevant
#
#     mkMerMod(rho = rho,
#             opt = optimizerOutput,
#             reTrms = parsedFormula$reTrms,
#             fr = parsedFormula$fr)
#
# }
#
# system.time(jm1 <- jmer(y ~ 1 + service + (1|s) + (1|d) + (1|dept), InstEval, REML=FALSE))
