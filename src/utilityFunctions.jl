function doFDR(pvalues,FDR=0.1)
    tri = sortperm(pvalues)
    m = length(pvalues)
    iBH = findall([0;pvalues[tri]].<= collect(0:m)/m*FDR)
    iBH = iBH[length(iBH)] # the last
    if (iBH<=m)
        H0 = tri[iBH:m]
    else H0 = Int64[]
    end
    H0
end

getChildren(pixIdx,infoDen,dataProj,dataVar,thrs,toCluster) = getChildren(CPU1(),pixIdx,infoDen,dataProj,dataVar,thrs,toCluster)

function getChildren(resource::CPU1,pixIdx,infoDen,dataProj,dataVar,thrs,toCluster)

    if (length(infoDen[pixIdx]) == 0)
        return nothing
    end
    ## the projection of the original dynamic of pix.idx
    pixOrg = dataProj[:,pixIdx]
    ## the projection of the denoised dynamic of pixIdx
    pixDen = infoDen[pixIdx]["Px"]
    ## the length of pix.idx neighborhood
    pixLen = length(infoDen[pixIdx]["Vx"])

    ## initialize the children of pix.idx to its neighbors
    children = getActive(infoDen[pixIdx]["Vx"],toCluster)

    i = 1
    while (i < length(children))
        i = i+1
        infoChild = infoDen[children[i]]
        varChild = dataVar[children[i]]
        ## test if denoised version are time coherent
        if multitestH0(infoChild["Px"]-pixOrg,varChild/length(infoChild["Vx"]) + dataVar[pixIdx],thrs)[1]
            ## Time coherent : we add non existing neighbors to children
            children = [children;setdiff(getActive(infoChild["Vx"],toCluster),children)]
        else
            ## not time coherent : remove from children
            deleteat!(children,i)
            i = i-1
        end
    end
    children::Array{Int64}
end

function getActive(indexes,toCluster)
    indexes[toCluster[indexes].==1]
end

checkClusterNew(center,centClusts,clusterDict,dataVar,quant,iter;pvalueMax=false,useFdr=true)=checkClusterNew(CPU1(),center,centClusts,clusterDict,dataVar,quant,iter,pvalueMax=pvalueMax,useFdr=useFdr)

function checkClusterNew(resource::CPU1,center,centClusts,clusterDict,dataVar,quant,iter;pvalueMax=false,useFdr=true)
    ## test new potential center against the centers from already build clusters
    ## if pvalue.max=T returns the index of the cluster giving the largest pvalue
    ## if pvalue.max=F returns the indexes of the clusters with pvalue corresponding to H0 (default)
    ## if use.fdr=TRUE (default) correct multiplicity with False Discovery Rate otherwise use Bonferroni correction
    sizeto = length(centClusts)


    if length(clusterDict)==0
        if !pvalueMax
            return([])
        else
            return(false)
        end
    end

    centVar = mean(dataVar[centClusts])/sizeto
    clusterCenters = Array{Float64}(undef,(length(center),length(clusterDict)))
    clusterVar = Array{Float64}(undef,length(clusterDict))
    clusterSizes = Array{Float64}(undef,length(clusterDict))
    for (i,x) in enumerate(clusterDict)
        clusterCenters[:,i] = x["center"]
        clusterSizes[i] = length(x["cluster"])
        clusterVar[i] = mean(x["cluster"])/length(x["cluster"])
    end

    ## test thresholds with Bonferroni correction adapted to both partition number and number of existing clusters
    if (pvalueMax || useFdr)
        thrs = nothing
    else
        thrs = [quantile(Chisq(i),1-quant/length(clusterDict)) for i=exp2(0:iter)]
    end
    tmp = multitestH0(clusterCenters.-center[:],clusterVar .+ centVar,thrs)
    ## return pvalues if thrs==NULL and booleans otherwise

    if pvalueMax
        return(argmax(tmp))
    elseif useFdr
        return(doFDR(tmp))
    else
        return(find(tmp))
    end
end

robustMean(children,dataProj,dataVar)=robustMean(CPU1(),children,dataProj,dataVar)

function robustMean(resource::CPU1,children,dataProj,dataVar)
    ## Build a robust 1-mean for a children list
    ## Initialize with the dynamic average over all children
    childProj = dataProj[:,children]
    cluster = children
    center = Inf
    i = 0
    iMax = 1000
    cont = true
    ## repeat until center does not change anymore
    while (cont && i<iMax)
        i = i+1
        ## compute the new cluster center
        centerNew = mean(dataProj[:,cluster],dims=2)
        ## test coherence between cluster center and children dynamics
        pvalues = multitestH0(childProj.-centerNew,dataVar[children].+mean(dataVar[cluster])/length(cluster),nothing)
        #println(pvalues)
        ## the new cluster
        cluster = children[doFDR(pvalues)]
        ## check if the cluster center has changed or not
        cont = (length(cluster)>0) && (mean((centerNew .- center).^2) > 0.0001)
        ## update
        center = centerNew
    end
    outliers=setdiff(children,cluster)

    if i==iMax
        println("stop RobustMean without stabilization")
    end

    Dict("cluster" => cluster,"center" => center,"outliers" => outliers,"OK" => (i==iMax))
end
