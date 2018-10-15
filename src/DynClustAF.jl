__precompile__()

module DynClustAF

using ComputationalResources, DynClust, ArrayFire, Distributions, StatsBase, ProgressMeter

function DynClust.runDenoising(resource::ArrayFireLibs,dataArray,dataMask,dataVar,alpha=0.05,maskSize=nothing)
    ballSize = cumsum([1;4;8;16;36;92;212;477])

    dim = size(dataArray)
    ndim = ndims(dataArray)
    ntime = dim[ndim]
    coord = dim[1:(ndim-1)]
    #coord = size(dataArray)[1:ndim-1]
    fullLength = prod(coord)
    dataMaskInd = (LinearIndices(dataMask))[findall(dataMask)]
    nvox = length(dataMaskInd)
    dataArray = copy(transpose(reshape(dataArray,(fullLength,ntime))[dataMaskInd,:]))
    iter = floor(Integer,log2(ntime))-1
    Dmax = exp2(iter)
    from = round(Int,exp2(iter+1))
    quant = alpha/(iter+1)
    realIter = floor(Integer,log2(from-1))-1
    thrs = [quantile.(Chisq(i),1-quant) for i=exp2.(0:(realIter-1))]


    if maskSize==nothing
        p = (1000/fullLength)^(1/length(coord))
        maskSize =ceil.(Integer,p.*[coord...])
        println("Mask size : $maskSize")
    elseif maskSize=="full"
        maskSize = [coord...]
    end


    ### Define the projection matrix
    dataProj = Array{Float64}(undef,(from-1,nvox))
    ## Inialize with the finest partition
    ## locations of time indexes in the finest partition
    loc = ceil.(Integer,collect(1:ntime)*Dmax/ntime)
    ## lengths of the finest partition intervals // returned as a vector
    num = counts(loc,maximum(loc))
    ## storage locations
    to = from-1
    from = to-length(num)+1
    ## projections
    for i=1:nvox
        dataProj[from:to,i]=counts(loc,maximum(loc),Weights(dataArray[:,i]))
    end

    ## loop from finer to thicker partitions
    for idx=(iter-1):-1:0
        ## new lengths of the partition intervals
        numNew = num[1:2:length(num)]+num[2:2:length(num)]
        ## new storage locations
        toNew = from-1
        fromNew = toNew-length(numNew)+1
        ## new projections
        dataProj[fromNew:toNew,:] = dataProj[from:2:to,:]+dataProj[from+1:2:to,:]
        ## normalize old projections
        dataProj[from:to,:] = dataProj[from:to,:]./sqrt.(num)
        ## update
        num = numNew
        to = toNew
        from = fromNew
    end
    ## last normalization is not done in the loop
    dataProj[1,:] = dataProj[1,:]./sqrt.(num)


    ## ########################### Build geometry #############################
    ## matrix of all the 3D coordinates at column indexes of data.matrix corresponds to row indexes in dataCoord
    
    dataCoord = CartesianIndices(coord)
    idOne = one(dataCoord[1])
    idBox = CartesianIndex(maskSize...)
    idMax = dataCoord[end]
    #dataCoord = reshape(dataCoord,fullLength,ndim-1)
    #arrCoord = reshape(1:fullLength,coord)
    arrCoord = LinearIndices(coord)
    
    ## transforms the data into a matrix of projections
    nproj = size(dataProj,1)

    ### Analysis
    println(nvox," voxels")
    prog = Progress(nvox,1)
    resVisited = map(1:nvox) do pixIdx
        ## Find the neighboors of the current voxel pixIdx build the mask hypercube around pixIdx
        pixInd = dataMaskInd[pixIdx]
        #dCo = dataCoord[pixInd:pixInd,:].'
        dCo = dataCoord[pixInd]
        #inf = broadcast(max,dCo-maskSize,1)
        inf = max(dCo-idBox,idOne)
        sup = min(idMax,dCo+idBox)
        #mask = IntSet(arrCoord[map((x,y) -> x:y,inf,sup)...])
        mask = BitSet(arrCoord[UnitRange.(inf.I,sup.I)...])
        intersect!(mask,BitSet(dataMaskInd))
        maskIdx = [findfirst(isequal(x),dataMaskInd) for x=mask]
        ## test in mask voxels which are homogenous with pixIdx       
        goodPix = multitestH0(AFArray(dataProj[:,maskIdx].-dataProj[:,pixIdx]),AFArray(dataVar[maskIdx].+dataVar[pixIdx]),thrs)[:]
        neighborsInd = collect(mask)[goodPix]
        dist = vec([sum(abs2.(c.I)) for c in dataCoord[neighborsInd] .- dCo])
        neighborsInd = neighborsInd[sortperm(dist)]


        neighborsIdx = map(neighborsInd) do x
            findfirst(isequal(x),dataMaskInd)
        end
        ## projection of the dynamics of the neighbors
        neighborsProj = dataProj[:,neighborsIdx]

        ## Denoising procedure
        ## number of possible balls
        nV = sum(ballSize.<=length(neighborsIdx))
        ## Iv.neighb is a list of the neighbors in each ball
        #Iv.neighb <- list()
        ## Iv is the matrix of the projection estimates build over the successive balls
        iv = Array{Float64}(undef,(nproj,nV))
        ## data.varIv vector of the associated variances
        datavarIv = Array{Float64}(undef,nV)

        ## Initialize
        limits = kV = 1
        iv[:,kV] = neighborsProj[:,1]
        datavarIv[kV] = dataVar[pixIdx]

        while kV<nV
            kV+=1
            limitsNew=ballSize[kV]
            ringIdx=(limits+1):limitsNew
            jvTemp=mean(neighborsProj[:,ringIdx],2)
            dataVarJvKv = mean(dataVar[ringIdx])/length(ringIdx)
            ## test thresholds with Bonferroni correction adapted to both partition number and interior balls

            thrs = [quantile(Chisq(i),1-quant/(kV-1)) for i=exp2.(0:(realIter-1))]
            ## test time coherence
            testcoh = multitestH0(AFArray(iv[:,1:(kV-1)].-jvTemp),AFArray(datavarIv[1:(kV-1)] .+ dataVarJvKv),thrs)

            ## if no time coherence with previous estimates
            if !all(testcoh)
                kV = kV-1
                break
            end
            ## otherwise update projection estimates
            limits=limitsNew
            iv[:,kV] = mean(neighborsProj[:,1:limits],dims=2)
            datavarIv[kV] = dataVar[pixIdx]/limits
        end
        ## the denoised dynamics rescaled

        ix = mean(dataArray[:,neighborsIdx[1:limits]],dims=2)

        #### returns a Dict containing:
        #### 'Vx' a vector containing all the neighbors indexes used to build the denoised dynamic
        #### 'Ix' a vector containing the denoised dynamic
        #### 'Px' a vector containing the denoised projection
        #### 'Lx' a matrix containing the original coordinates of the neighbors in data.array
        #### 'Cx' a vector containing the original coordinates of the center
        next!(prog)
        Dict("Lx" => vec(dataCoord[neighborsInd[1:limits]]),"Cx"=>dataCoord[pixInd],"Px" => vec(iv[:,kV]),"Ix" =>vec(ix),"Vx"=>vec(neighborsIdx[1:limits]))
    end
    Dict("infoDen" => resVisited,"dataProj" => dataProj, "var"=> dataVar)
end


function multitestH0(projMatrix::AFArray,dataVar,thrs::Nothing)
    projMatrix = copy(transpose(projMatrix))./dataVar
    kMax = floor(Integer,log2(size(projMatrix,2)))-1
    inds = round.(Int,exp2.(0:kMax))
    #pval = zeros(nc)
    projMatrix = abs2.(projMatrix)
    cs = Array(cumsum(projMatrix,2))[:,(inds[2:end].-1)]
    pval = maximum(cdf.(Chisq.(inds[1:(end-1)]),hcat(cs[:,inds[2]-1],diff(cs,2)).'),1)
    #pval = maximum(cdf.(Chisq.(inds[1:(end-1)]),diff(cs,2).'),1)
    return((1-pval)[:])
end

function multitestH0(projMatrix::AFArray,dataVar,thrs::Array{Float64,1})
    projMatrix = copy(transpose(projMatrix))./dataVar
    kMax = floor(Integer,log2(size(projMatrix,2)))-1
    inds = round.(Int,exp2.(0:kMax))
    projMatrix = abs2.(projMatrix)
    cs = Array(cumsum(projMatrix,2))[:,inds[2:end].-1]
    norm2proj = hcat(cs[:,inds[2]-1],diff(cs,2)).'
    #norm2proj = diff(cs,2)
    test = all(norm2proj.<=thrs,dims=1)
    return(test[:])
end


function getChildren(resource::ArrayFireLibs,pixIdx,infoDen,dataProj,dataVar,thrs,toCluster)

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
    children = DynClust.getActive(infoDen[pixIdx]["Vx"],toCluster)

    i = 1
    while (i < length(children))
        i = i+1
        infoChild = infoDen[children[i]]
        varChild = dataVar[children[i]]
        ## test if denoised version are time coherent
        if multitestH0(AFArray(infoChild["Px"]-pixOrg),varChild/length(infoChild["Vx"]) + dataVar[pixIdx],thrs)[1]
            ## Time coherent : we add non existing neighbors to children
            children = [children;setdiff(DynClust.getActive(infoChild["Vx"],toCluster),children)]
        else
            ## not time coherent : remove from children
            deleteat!(children,i)
            i = i-1
        end
    end
    children::Array{Int64}
end

function DynClust.runClustering(resource::ArrayFireLibs,dataArray,dataMask,denois;minSize=1,alpha=0.05,doChildrenFirst=false)

    dim = size(dataArray)
    ndim = ndims(dataArray)
    ntime = dim[ndim]
    coord = dim[1:(ndim-1)]
    fullLength = prod(coord)
    dataMaskInd = find(dataMask)
    nvox = length(dataMaskInd)
    dataArray = (reshape(dataArray,(fullLength,ntime))[dataMaskInd,:]).'
    iter = floor(Integer,log2(ntime))-1
    from = round(Int,exp2(iter+1))
    realIter = floor(Integer,log2(from-1))-1
    quant = alpha/(iter+1)
    thrs = [quantile(Chisq(i),1-quant) for i=exp2.(0:(realIter-1))]

    infoDen = denois["infoDen"]
    dataProj = denois["dataProj"]
    dataVar = denois["var"]

    toCluster = ones(Int64,length(infoDen))

    if doChildrenFirst
         ## Build the children list of all denoised voxels
        childrenDict = map(1:length(infoDen)) do idx
            getChildren(ArrayFireLibs(),idx,infoDen,dataProj,dataVar,thrs,toCluster)
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
            childrenOk = getChildren(ArrayFireLibs(),idCl,infoDen,dataProj,dataVar,thrs,toCluster)
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
            ind = checkClusterNew(ArrayFireLibs(),dataProj[:,idCl],idCl,clusterDict,dataVar,quant,realIter,pvalueMax=true)

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
           newCluster = robustMean(ArrayFireLibs(),childrenOk,dataProj,dataVar)
            #println(length(newCluster["cluster"]))
           check = checkClusterNew(ArrayFireLibs(),newCluster["center"],newCluster["cluster"],clusterDict,dataVar,quant,realIter,pvalueMax=false,useFdr=true)
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
            ind = checkClusterNew(ArrayFireLibs(),dataProj[:,idCl],idCl,clusterDict,dataVar,quant,realIter,pvalueMax=true)
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

function checkClusterNew(resource::ArrayFireLibs,center,centClusts,clusterDict,dataVar,quant,iter;pvalueMax=false,useFdr=true)
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
    clusterCenters = Array{Float64}((length(center),length(clusterDict)))
    clusterVar = Array{Float64}(length(clusterDict))
    clusterSizes = Array{Float64}(length(clusterDict))
    for (i,x) in enumerate(clusterDict)
        clusterCenters[:,i] = x["center"]
        clusterSizes[i] = length(x["cluster"])
        clusterVar[i] = mean(x["cluster"])/length(x["cluster"])
    end

    ## test thresholds with Bonferroni correction adapted to both partition number and number of existing clusters
    if (pvalueMax || useFdr)
        thrs = nothing
    else
        thrs = [quantile(Chisq(i),1-quant/length(clusterDict)) for i=exp2(0:(iter-1))]
    end
    tmp = multitestH0(AFArray(clusterCenters.-center[:]),AFArray(clusterVar+centVar),thrs)
    ## return pvalues if thrs==NULL and booleans otherwise

    if pvalueMax
        return(indmax(tmp))
    elseif useFdr
        return(DynClust.doFDR(tmp))
    else
        return(find(tmp))
    end
end

function robustMean(resource::ArrayFireLibs,children,dataProj,dataVar)
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
        centerNew = mean(dataProj[:,cluster],2)
        ## test coherence between cluster center and children dynamics
        pvalues = multitestH0(AFArray(childProj.-centerNew),AFArray(dataVar[children].+mean(dataVar[cluster])/length(cluster)),nothing)
        ## the new cluster
        cluster = children[DynClust.doFDR(pvalues)]
        ## check if the cluster center has changed or not
        cont = (length(cluster)>0) && (mean((centerNew-center).^2) > 0.0001)
        ## update
        center = centerNew
    end
    outliers=setdiff(children,cluster)

    if i==iMax
        println("stop RobustMean without stabilization")
    end

    Dict("cluster" => cluster,"center" => Array(center),"outliers" => outliers,"OK" => (i==iMax))
end


end
