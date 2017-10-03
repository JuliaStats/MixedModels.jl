using Weave

weave("./jmd/constructors.jmd", doctype="github", plotlib="Gadfly",
      fig_path="./assets/", fig_ext=".svg", out_path="./src/")

weave("./jmd/extractors.jmd", doctype="github", plotlib="Gadfly",
      fig_path="./assets/", fig_ext=".svg", out_path="./src/")

weave("./jmd/bootstrap.jmd", doctype="github", plotlib="Gadfly",
      fig_path="./assets/", fig_ext=".svg", out_path="./src/")
