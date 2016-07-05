"""

    runDenoising(dataArray,dataMask,dataVar,depth=1,alpha=0.05,maskSize=nothing)


Runs the denoising step.
"""
function runDenoising(dataArray,dataMask,dataVar,depth=1,alpha=0.05,maskSize=nothing)
    ballSize = cumsum([1;4;8;16;36;92;212;477])

    dim = size(dataArray)
    ndim = ndims(dataArray)
    ntime = dim[ndim]
    coord = dim[1:(ndim-1)]
    fullLength = prod(coord)
    dataMaskInd = find(dataMask)
    nvox = length(dataMaskInd)
    dataArray = (reshape(dataArray,(fullLength,ntime))[dataMaskInd,:]).'
    iter = floor(Integer,log2(ntime))-1
    Dmax = exp2(iter)
    from = round(Int,exp2(iter+1))
    quant = alpha/(iter+1)
    thrs = [quantile(Chisq(i),1-quant) for i=exp2(0:iter)]


    if maskSize==nothing
        p = (1000/fullLength)^(1/length(coord))
        maskSize =ceil(Integer,p.*[coord...])
        println("Mask size : $maskSize")
    elseif maskSize=="full"
        maskSize = [coord...]
    end


    ### Define the projection matrix
    dataProj = Array(Float64,(from-1,nvox))
    ## Inialize with the finest partition
    ## locations of time indexes in the finest partition
    loc = ceil(Integer,collect(1:ntime)*Dmax/ntime)
    ## lengths of the finest partition intervals // returned as a vector
    num = counts(loc,maximum(loc))
    ## storage locations
    to = from-1
    from = to-length(num)+1
    ## projections
    for i=1:nvox
        dataProj[from:to,i]=counts(loc,maximum(loc),WeightVec(dataArray[:,i]))
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
        dataProj[from:to,:] = dataProj[from:to,:]./sqrt(num)
        ## update
        num = numNew
        to = toNew
        from = fromNew
    end
    ## last normalization is not done in the loop
    dataProj[1,:] = dataProj[1,:]./sqrt(num)


    ## ########################### Build geometry #############################
    ## matrix of all the 3D coordinates at column indexes of data.matrix corresponds to row indexes in dataCoord
    dataCoord = ind2sub(coord,1:fullLength)
    dataCoord = reshape(vcat(dataCoord...),fullLength,ndim-1)
    arrCoord = reshape(1:fullLength,coord)
    ## transforms the data into a matrix of projections
    nproj = size(dataProj,1)

    ### Analysis
    resVisited = pmap(1:nvox) do pixIdx
        ## Find the neighboors of the current voxel pixIdx build the mask hypercube around pixIdx
        pixInd = dataMaskInd[pixIdx]
        dCo = dataCoord[pixInd,:].'
        inf = broadcast(max,dCo-maskSize,1)
        sup = min([coord...],dCo+maskSize)
        mask = IntSet(arrCoord[map((x,y) -> x:y,inf,sup)...])
        intersect!(mask,IntSet(dataMaskInd))
        maskIdx = [findfirst(dataMaskInd,x) for x=mask]
        ## test in mask voxels which are homogenous with pixIdx
        goodPix = multitestH0(dataProj[:,maskIdx].-dataProj[:,pixIdx],dataVar[maskIdx].+dataVar[pixIdx],thrs)[:]
        neighborsInd = collect(mask)[goodPix]
        dist = vec(sumabs2(dataCoord[neighborsInd,:].-dCo.',2))
        neighborsInd = neighborsInd[sortperm(dist)]


        neighborsIdx = map(neighborsInd) do x
            findfirst(dataMaskInd,x)
        end
        ## projection of the dynamics of the neighbors
        neighborsProj = dataProj[:,neighborsIdx]

        ## Denoising procedure
        ## number of possible balls
        nV = sum(ballSize.<=length(neighborsIdx))
        ## Iv.neighb is a list of the neighbors in each ball
        #Iv.neighb <- list()
        ## Iv is the matrix of the projection estimates build over the successive balls
        iv = Array(Float64,(nproj,nV))
        ## data.varIv vector of the associated variances
        datavarIv = Array(Float64,nV)

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

            thrs = [quantile(Chisq(i),1-quant/(kV-1)) for i=exp2(0:iter)]
            ## test time coherence
            testcoh = multitestH0(iv[:,1:(kV-1)].-jvTemp,datavarIv[1:(kV-1)]+dataVarJvKv,thrs)

            ## if no time coherence with previous estimates
            if !all(testcoh)
                kV = kV-1
                break
            end
            ## otherwise update projection estimates
            limits=limitsNew
            iv[:,kV] = mean(neighborsProj[:,1:limits],2)
            datavarIv[kV] = dataVar[pixIdx]/limits
        end
        ## the denoised dynamics rescaled

        ix = mean(dataArray[:,neighborsIdx[1:limits]],2)

        #### returns a Dict containing:
        #### 'Vx' a vector containing all the neighbors indexes used to build the denoised dynamic
        #### 'Ix' a vector containing the denoised dynamic
        #### 'Px' a vector containing the denoised projection
        #### 'Lx' a matrix containing the original coordinates of the neighbors in data.array
        #### 'Cx' a vector containing the original coordinates of the center
        Dict("Lx" => vec(dataCoord[neighborsInd[1:limits],:]),"Cx"=>vec(dataCoord[pixInd,:]),"Px" => vec(iv[:,kV]),"Ix" =>vec(ix),"Vx"=>vec(neighborsIdx[1:limits]))
    end
    Dict("infoDen" => resVisited,"dataProj" => dataProj, "var"=> dataVar)
end


function getDenoisingResults(dataArray,denoisDicArr)
    for i in 1:length(denoisDicArr["infoDen"])
        tmp = denoisDicArr["infoDen"][i]
        ## get the spatial coordinates of the center
        mid = tmp["Cx"]
        ## replace the dynamics in dataArray by its denoised version
        dataArray[mid...,:] = tmp["Ix"]
    end
    dataArray
end
