__precompile__(true)

module DynClust
using StatsBase
using Distributions


include("utilityFunctions.jl")
include("multiTest.jl")
include("denoising.jl")
include("clustering.jl")

export  runDenoising,
        getDenoisingResults,
        runClustering,
        getClusteringResults


end # module
