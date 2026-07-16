using QuditClifford
using Documenter

DocMeta.setdocmeta!(QuditClifford, :DocTestSetup, :(using QuditClifford); recursive=true)

makedocs(;
    modules=[QuditClifford],
    authors="Malte Pütz",
    sitename="QuditClifford.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://maltepuetz.github.io/QuditClifford.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Examples" => "examples.md",
        "Representation and Conventions" => "conventions.md",
        "Measurements and Expectations" => "measurements.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/maltepuetz/QuditClifford.jl",
    devbranch="main",
)
