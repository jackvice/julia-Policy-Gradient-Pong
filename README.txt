These Julia source files contain a Julia implementation of policy gradient for OpenAI Gym Pong and Bipedal Walker 2D.  Pong solution is based on Andrej Karpathy's solution:
http://karpathy.github.io/2016/05/31/rl/

Walker2D currenly only learns "not to fall".  Needs a better reward function.  Walker2D reads the renderPrint.txt file at some interval.  renderPrint.txt has three integers.  The first integer is 1 for render and 0 not.  The second is 1 for print to screen more data and 0 for not and the third integer is 1 for go on policy and 0 for include some random exploration. 


Dependencies are:
using Dates
using DelimitedFiles
import Gym
import Random

to run Pong:
include("pgPong.jl")
main()



to run 2DWalker:
include("pg2Dwalker.jl")
main()

