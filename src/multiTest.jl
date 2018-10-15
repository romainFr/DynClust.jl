function multitestH0(projMatrix::Array{Float64,2},dataVar::Array{Float64,1},thrs::Nothing)
    projMatrix = transpose(projMatrix)./dataVar
    (nc,nr) = size(projMatrix)
    kMax = floor(Integer,log2(nr))-1
    inds = round.(Int,exp2.(0:kMax))
    pval = zeros(nc)
    projMatrix = abs2.(projMatrix)
    cs = Array(cumsum(projMatrix,dims=2))[:,(inds[2:end].-1)]
    pval = maximum(cdf.(Chisq.(inds[1:(end-1)]),transpose(hcat(cs[:,inds[2]-1],diff(cs,2)))),dims=1)
    
   # for (inds,inds2) in zip(inds[1:end-1],inds[2:end] .- 1)
     
    #    norm2proj = vec(sum(projMatrix[:,inds:inds2],2))
     
    #    @inbounds   pval = max.(pval,cdf.(Chisq(inds),norm2proj))
    #end
    return((1-pval)[:])
end

function multitestH0(projMatrix::Array{Float64},dataVar,thrs::Array{Float64,1})
    projMatrix = transpose(projMatrix)./dataVar
    (nc,nr) = size(projMatrix)
    kMax = floor(Integer,log2(nr))-1
    inds = round.(Int,exp2.(0:kMax))
    #@show inds
   # test = trues(nc)
    projMatrix = abs2.(projMatrix)
    cs = Array(cumsum(projMatrix,dims=2))[:,inds[2:end].-1]
    norm2proj = permutedims(hcat(cs[:,inds[2]-1],diff(cs,dims=2)))
    #@show size(cs)
    #norm2proj = diff(cs,2)
    #@show size(norm2proj)
    #@show length(thrs)
    test = all(norm2proj.<=thrs,dims = 1)
    
    #for (inds,inds2,i) in zip(inds[1:end-1],inds[2:end] .- 1,collect(1:length(inds) .- 1))
       
    #    norm2proj = vec(sum(projMatrix[:,inds:inds2],2))
     
    #    @inbounds test = test .& (norm2proj.<=thrs[i])
    #end
    return(test[:])
end

