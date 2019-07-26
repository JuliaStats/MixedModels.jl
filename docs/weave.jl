using Weave

weavit(fnm::AbstractString) =
    weave(joinpath("jmd", fnm), doctype="github", fig_path="./assets", fig_ext=".svg",
    out_path="./src")

weavit("constructors.jmd")
weavit("optimization.jmd")
weavit("bootstrap.jmd")
#weavit("SimpleLMM.jmd")
#weavit("MultipleTerms.jmd")
#weavit("SingularCovariance.jmd")
#weavit("SubjectItem.jmd")
weavit("GaussHermite.jmd")
