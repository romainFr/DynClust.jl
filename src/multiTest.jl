function multitestH0(projMatrix,dataVar,thrs::Void)
    projMatrix = projMatrix./dataVar.'
    (nr,nc) = size(projMatrix)
    kMax = floor(Int64,log2(nr))-1
    inds = round(Int64,exp2(0:kMax))
    pval = zeros(nc)
    for (inds,inds2) in zip(inds[1:end-1],inds[2:end]-1)
        #norm2proj = squeeze(sumabs2(projMatrix[:,inds:inds2],2),2)
        norm2proj = vec(sumabs2(projMatrix[inds:inds2,:],1))
        #sumabs2!(norm2proj,projMatrix[:,inds:inds2])
        #   @inbounds   broadcast!(max,pval,pval,cdf(Chisq(inds),norm2proj))
        @inbounds   pval = max(pval,cdf(Chisq(inds),norm2proj))
    end
    return(1-pval)
end

function multitestH0(projMatrix,dataVar,thrs::Array{Float64,1})
    projMatrix = projMatrix./dataVar.'
    (nr,nc) = size(projMatrix)
    kMax = floor(Int64,log2(nr))-1
    inds = round(Int64,exp2(0:kMax))
    test = trues(nc)
    for (inds,inds2,i) in zip(inds[1:end-1],inds[2:end]-1,collect(1:length(inds)-1))
        #norm2proj = squeeze(sumabs2(projMatrix[:,inds:inds2],2),2)
        norm2proj = vec(sumabs2(projMatrix[inds:inds2,:],1))
        # @inbounds   broadcast!(&,test,test,norm2proj.<=thrs[i])
        @inbounds test = test & (norm2proj.<=thrs[i])
    end
    return(test)
end
