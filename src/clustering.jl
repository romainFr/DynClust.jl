runClustering(dataArray,dataMask,denois;minSize=1,alpha=0.05,doChildrenFirst=false)=runClustering(CPU1(),dataArray,dataMask,denois,minSize=minSize,alpha=alpha,doChildrenFirst=doChildrenFirst)

function runClustering(resource::CPU1,dataArray,dataMask,denois;minSize=1,alpha=0.05,doChildrenFirst=false)

    dim = size(dataArray)
    ndim = ndims(dataArray)
    ntime = dim[ndim]
    coord = dim[1:(ndim-1)]
    fullLength = prod(coord)
    dataMaskInd = find(dataMask)
    nvox = length(dataMaskInd)
    dataArray = (reshape(dataArray,(fullLength,ntime))[dataMaskInd,:]).'
    iter = floor(Integer,log2(ntime))-1

    quant = alpha/(iter+1)
    thrs = [quantile(Chisq(i),1-quant) for i=exp2(0:iter)]

    infoDen = denois["infoDen"]
    dataProj = denois["dataProj"]
    dataVar = denois["var"]

    toCluster = ones(Int64,length(infoDen))

    if doChildrenFirst
         ## Build the children list of all denoised voxels
        childrenDict = pmap(1:length(infoDen)) do idx
            getChildren(idx,infoDen,dataProj,dataVar,thrs,toCluster)
        end
    end



    badVoxels = Int64[]
    clusterDict = Dict[]
    actualMinSize = Inf

    while any(toCluster.==1)
        idToCluster = find(toCluster)
        println(length(idToCluster), "---", actualMinSize)
        ## Find in the remaining voxels to clusterize
        ## that with the largest list of children (if doChildrenFirst=true) or neighbors (otherwise)
        ## remaining to clusterize
        if doChildrenFirst
            sizeto = map(childrenDict[idToCluster]) do x
                sum(toCluster[x])
            end
        else
            sizeto = map(infoDen[idToCluster]) do x
                sum(toCluster[x["Vx"]])
            end
        end

        idMax = indmax(sizeto)
        actualMinSize = sizeto[idMax]
        if actualMinSize<minSize
            break
        end
        idCl = idToCluster[idMax]

        ## build cluster from voxel defined by id.tocluster
        ## find the active children (still in the list of voxels to clusterize) of id.tocluster
        if doChildrenFirst
            childrenOk = childrenDict[idCl]
        else
            childrenOk = getChildren(idCl,infoDen,dataProj,dataVar,thrs,toCluster)
        end
        ## children are active children if toCluster[children] is true
        if length(childrenOk)==0
            toCluster[idCl] = 0
            println("nothing to do")
            continue
        end

        inChildrenLength = length(childrenOk)
        if inChildrenLength <= minSize
            ## not enough children, we just add voxel to closest cluster
            #addToClosestCluster(idCl,toCluster,clusterDict,badVoxels,dataProj,dataVar,quant,iter)
            toCluster[idCl] = 0
            ## add voxel to the closest cluster i.e. with largest p-value
            ind = checkClusterNew(dataProj[:,idCl],idCl,clusterDict,dataVar,quant,iter,pvalueMax=true)

            if !isa(ind,Bool)
                clusterDict[ind]["cluster"] = [clusterDict[ind]["cluster"];idCl]
                clusterDict[ind]["center"] = mean(dataProj[:,clusterDict[ind]["cluster"]],2)
            end
            badVoxels = [badVoxels;idCl]
            continue
        end

        check = 1
        connectedClusters = Dict[]
        newCluster = Dict[]

        while length(check)>0
           newCluster = robustMean(childrenOk,dataProj,dataVar)
            #println(length(newCluster["cluster"]))
           check = checkClusterNew(newCluster["center"],newCluster["cluster"],clusterDict,dataVar,quant,iter,pvalueMax=false,useFdr=true)
           # println(check)
           if length(check)>0
           ## at least, one existing cluster is coherent with the new one
           ## get the connected clusters
               connectedClusters = [connectedClusters;clusterDict[check]]
           ## get the connected voxels id
               #println(length(clusterDict[check][1]["cluster"]))
               connectedIdx = mapreduce((x) ->  x["cluster"],vcat,[],clusterDict[check])
               #println(length(connectedIdx))
           ## remove the connected clusters from the cluster list
               deleteat!(clusterDict,sort(check))
           ## add the connected in the neighbors of idx.tocluster
               childrenOk = unique([childrenOk;connectedIdx])
           end
        end

        if length(newCluster["outliers"])<inChildrenLength
            ## we found a new cluster !!!
            ## remove the new cluster from pixel to clusterize
            toCluster[newCluster["cluster"]] = 0
            toCluster[newCluster["outliers"]] = 1
            ## update cluster list
            clusterDict = [clusterDict;Dict("cluster"=>newCluster["cluster"],"center"=>newCluster["center"])]
            #println(length(clusterDict))
        else
            ## TOO MANY OUTLIERS: we do not build new cluster
            ## Return cluster list to its original state
            ## Add id.cluster to the closest cluster
            ## return cluster list in its original state
            clusterDict = [clusterDict;connectedClusters]
            ## add voxel to closest cluster
            ## mark voxel as already clusterize
            toCluster[idCl] = 0

            ## add voxel to the closest cluster i.e. with largest p-value
            ind = checkClusterNew(dataProj[:,idCl],idCl,clusterDict,dataVar,quant,iter,pvalueMax=true)
            if !isa(ind,Bool)
                clusterDict[ind]["cluster"] = [clusterDict[ind]["cluster"];idCl]
                clusterDict[ind]["center"] = mean(dataProj[:,clusterDict[ind]["cluster"]],2)
            end
            badVoxels = [badVoxels;idCl]
        end


    end
    ## sort the cluster list in decreasing order
    sort!(clusterDict,by = ((x) -> length(x["cluster"])),rev=true)
    println(length(clusterDict))
    ## compute the cluster centers
    for i in 1:length(clusterDict)
        clusterDict[i]["center"]=mean(dataArray[:,clusterDict[i]["cluster"]],2)
    end

    ## extract centers and clusters
    centers = map((x) -> x["center"],clusterDict)
    clusters = map((x) ->  x["cluster"],clusterDict)

    return(Dict("clusters" => clusters, "centers" => centers, "badVoxels" => badVoxels))
end

function getClusteringResults(dataArray,denois,clust)
    clustMap = 0*Array{Int64}(size(dataArray)[1:(end-1)])

    for i in 1:length(clust["clusters"])
        for j in clust["clusters"][i]
            tmp = denois["infoDen"][j]
            if tmp == nothing
                continue
            end
            mid = tmp["Cx"]
            dataArray[mid...,:] = clust["centers"][i]
            clustMap[mid...] = i
        end
    end
    Dict("clustArray" => dataArray,"clustMap"=>clustMap,"clustCenter"=>clust["centers"])
end
