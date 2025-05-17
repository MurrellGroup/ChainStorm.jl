using ChainStorm
using Documenter

DocMeta.setdocmeta!(ChainStorm, :DocTestSetup, :(using ChainStorm); recursive=true)

makedocs(;
    modules=[ChainStorm],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="ChainStorm.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/ChainStorm.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/ChainStorm.jl",
    devbranch="main",
)
