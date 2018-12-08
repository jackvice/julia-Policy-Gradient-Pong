using Dates
using DelimitedFiles
import Gym
import Random


function main(;
        renderPeriod = 2,
        #episodes = 4,  # max episodes played
        render = false,
        randSeed = 17,
	btype = Array{Float32}, #no gpu for now
	atype = Array{Float32} #no gpu for now
        #atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}, #(C)
    )
    decayRate = 0.99 # decay Rate for RMSProp leaky sum of grad^2
    batchSize = 10  # every how many episodes to do a param update?
    writeWeightsHiddenFile = 1000
    learningRate = 1e-3 #
    gamma = 0.99 # discount factor for reward
    inputDim = 80 * 80
    startUnixTime = time()
    hiddenSize = 200
    rewardSum = 0
    episodeNum = 0
    historyObservations = atype(undef, 6400, 0) #
    historyRewards = btype(undef, 1, 0) #
    historyPredict = btype(undef, 1, 0) #
    historyWeights = atype(undef, 2, 0) #
    tempGradientSum = btype[ zeros(Float32, 200,6400), zeros(Float32,200,)]
    historyLossGradient = btype(undef, 1, 0) #
    historyHidden = atype(undef, 200, 0) #
    weights = atype[ randn(Float32,hiddenSize,inputDim)/sqrt(inputDim),
                     randn(Float32,hiddenSize,)/sqrt(hiddenSize)]

    expectationGsquared = atype[zeros(200,6400), zeros(200)]
    gDict = atype[zeros(200,6400),zeros(200,6400)]
    runningReward = false
    
    env = Gym.GymEnv("Pong-v0")# 
    if randSeed > 0 # This if block is same as previous line but clearer
        Random.seed!(randSeed)
        Gym.seed!(env, randSeed)
    end
    observation = Gym.reset!(env)

    oldTime = time()
    maxTime = 0
    previousFrame = convert(atype,zeros(inputDim))
    eStart = time()
    while true
	render && Gym.render(env)
         
	currentFrame = preprocess(observation,atype)

	diffFrame = currentFrame - previousFrame
        previousFrame = copy(currentFrame)

	(hiddenValues, upProb) = predict(weights, diffFrame)

	historyObservations = [ historyObservations diffFrame ] #append
	historyHidden = [ historyHidden hiddenValues ]

	action = getActionFromProb(upProb)

	observation, reward, done, info = Gym.step!(env, action) # one output

	rewardSum += reward
	historyRewards = [ historyRewards reward ]
	if action == 2
	    fakeLabel = 1
	else
	    fakeLabel = 0
	end
	lossGradient = fakeLabel - upProb

	historyLossGradient = [ historyLossGradient  lossGradient ]

	if done # episode finished
	    episodeNum += 1
	    gradientLogDiscounted = DiscountWithRewards(historyLossGradient,
                                                        historyRewards, gamma)
	    gradientLogDiscounted = convert(atype, gradientLogDiscounted)
	    gradient = gradientCalc(gradientLogDiscounted, historyHidden,
				    historyObservations, weights)
	    for i = 1:2
		tempGradientSum[i] += convert(btype,gradient[i]')
	    end

	    if episodeNum % batchSize == 0
		println("updating weights")
		weightsUpdate(weights,learningRate, decayRate, expectationGsquared, tempGradientSum)

            end

            writeDataToFile(episodeNum, convert(Array{Float32},weights[1]),
                            convert(Array{Float32},weights[2]),
                            rewardSum, runningReward, startUnixTime)

	    historyObservations = atype(undef, 6400, 0) #
	    historyHidden = atype(undef, 200, 0) #
	    historyLossGradient = btype(undef, 1, 0) #
	    historyRewards = btype(undef, 1, 0) #
	    observation = Gym.reset!(env)
	    
	    if runningReward == false
		runningReward = rewardSum
	    else
		runningReward = runningReward * 0.99 + rewardSum * 0.01
	    end
            
	    println("Episode #:",episodeNum,",  Resetting, game score: ", rewardSum, ",  Running mean: ", runningReward)
	    rewardSum = 0
	    previousFrame = convert(atype,zeros(inputDim))
	    eStart = time()    
	end
        
    end
    render && Gym.close!(env)
    
end


function printTime(str,oldTime, maxTime)
    diffTime = time() - oldTime
    if diffTime > 1
        println("time of ",str, diffTime)
    end 
    return (time(), maxTime)
end

function printTimeold(str,oldTime, maxTime)
    diffTime = time() - oldTime
    if diffTime > maxTime
        println("time of ",str,diffTime)
        return (time(), diffTime)
    else
        return (time(), maxTime)
    end
end



function writeDataToFile(e, weights1,weights2, reward, runningReward,startTime)
    writeWeightsPeriod = 500
    dataFileName = "dataE3.csv"
    dateStr = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    weightsFileName = string("E-",string(e), "weights", dateStr, ".txt" ) 

    if e % writeWeightsPeriod == 0
        open(weightsFileName, "w") do io
            writedlm(io, [weights1,weights2], ',')
        end
    end
    open(dataFileName, "a") do io
        writedlm(io, [dateStr (time()-startTime) e reward runningReward], ',')
      
    end
end

#fix to knet in here.
function weightsUpdate(weights,learningRate, decayRate, expectationGsquared, gBatchSum)
    e = 1e-5
    for i = 1:2
	tempGradient = gBatchSum[i]
	expectationGsquared[i] = decayRate * convert(Array{Float32},expectationGsquared[i]) +
            ((1-decayRate) * tempGradient.^2)

	z1 = convert(Array{Float32},(learningRate * tempGradient)) # make knet later
	z2 = convert(Array{Float32},(sqrt.(expectationGsquared[i] .+ e)))
	z3 = z1 ./ z2
        z3 = convert(KnetArray{Float32},z3)
        if i == 2
            weights[i] = weights[i] + vec(z3)
        else
            weights[i] += z3
        end
	gBatchSum[i] = zeros(Float32, size(weights[i])) #zero out the batch gradient buffer 
    end
end

function gradientCalc(gradientLogDiscounted, historyHidden, historyObservations, weights)
    deltaLog = gradientLogDiscounted
    DCost_DWeight2 = (deltaLog * historyHidden')
    deltaLog2 = deltaLog.*weights[2]
    deltaLog2 = relu.(deltaLog2)
    DCost_DWeight1 = (historyObservations * deltaLog2')
    return (DCost_DWeight1, DCost_DWeight2)
end

mean(x) = sum(x) / length(x)
std(z) = sqrt(mean(map(x -> (x - mean(z))^2, z)))

function DiscountWithRewards(historyLossGradient, historyRewards, gamma)
    discountEpisodeRewards = discountRewards(historyRewards,gamma)
    discountEpisodeRewards = discountEpisodeRewards .- mean(discountEpisodeRewards)
    discountEpisodeRewards = discountEpisodeRewards ./ std(discountEpisodeRewards)
    return (historyLossGradient .* discountEpisodeRewards)
end


function discountRewards(rewards, gamma)
    rewardsDiscounted = zeros(size(rewards))
    tempAdd = 0.0
    for i = length(rewards):-1:1
        if rewards[i] != 0.0
            tempAdd = 0.0
        end
        tempAdd = tempAdd * gamma + rewards[i]
        rewardsDiscounted[i] = tempAdd
    end
    return rewardsDiscounted
end

#take a probability and return a int action using random
function getActionFromProb(probAction)
    x = rand()
    if x < probAction
        return 2 #up
    else
    	return 3 #down
    end
end


sigmoid(z) = 1.0 ./ (1.0 .+ exp(-z))

relu(x) = x * (x > 0)

function predictCPU(weights, observation)
    hiddenLayerValues = weights[1] * observation
    hiddenLayerValues = relu.(hiddenLayerValues)
    outputLayerValues = dot(hiddenLayerValues, weights[2])
    outputLayerValues = sigmoid(outputLayerValues)
    return (hiddenLayerValues, outputLayerValues)
end

function predict(weights, observation)
    hiddenLayerValues = weights[1] * reshape(observation, 6400,1)
    hiddenLayerValues = relu.(hiddenLayerValues)
    outputLayerValues = reshape(weights[2], 1, 200) * hiddenLayerValues
    outputLayerValues = sigmoid(outputLayerValues[1])
    return (hiddenLayerValues, outputLayerValues)
end


# function downloaded from
#https://github.com/CarloLucibello/DeepRLexamples.jl
function preprocess(I, atype)
    I = I[36:195,:,:]
    I = I[1:2:end, 1:2:end, 1]
    I[I .== 144] .= 0
    I[I .== 109] .= 0
    I[I .!= 0] .= 1
    return convert(atype, vec(I))
end
