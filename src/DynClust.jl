__precompile__(true)

module DynClust
using StatsBase
using Distributions
using ComputationalResources
using Distributed
using ProgressMeter
using Images
using OffsetArrays

function __init__()
    # Enable `using` to load additional modules in this folder
    push!(LOAD_PATH, dirname(@__FILE__))
    # Now check for any resources that your package supports
    if haveresource(ArrayFireLibs)
        # User has indicated support for the ArrayFire libraries, so load your relevant code
        @eval using DynClustAF
    end
    # Put additional resource checks here
    # Don't forget to clean up!
    pop!(LOAD_PATH)
end

include("utilityFunctions.jl")
include("multiTest.jl")
include("denoising.jl")
include("clustering.jl")

export  runDenoising,
        getDenoisingResults,
        runClustering,
        getClusteringResults


end # module
