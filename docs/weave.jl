using Weave

weave("./jmd/constructors.jmd", doctype="github", plotlib="Gadfly",
      fig_ext=".svg", out_path="./src/")
