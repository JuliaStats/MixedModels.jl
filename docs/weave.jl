using Weave

isdir("./build/cache") || mkdir("./build/cache/")

weave("./jmd/SimpleLMM.jmd", doctype="github", plotlib="Gadfly",
       fig_path="./assets/", fig_ext=".svg", out_path="./src/",
       cache_path="./build/cache/", cache=:user)

weave("./jmd/constructors.jmd", doctype="github", plotlib="Gadfly",
      fig_path="./assets/", fig_ext=".svg", out_path="./src/")

weave("./jmd/extractors.jmd", doctype="github", plotlib="Gadfly",
      fig_path="./assets/", fig_ext=".svg", out_path="./src/")

weave("./jmd/bootstrap.jmd", doctype="github", plotlib="Gadfly",
      fig_path="./assets/", fig_ext=".svg", out_path="./src/")
