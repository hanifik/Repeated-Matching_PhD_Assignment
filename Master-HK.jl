using LinearAlgebra, JuMP, Ipopt, Random, Plots, CSV, DataFrames

###################################################################          ###################################################################
#                                                                  Question 3                                                                  #
###################################################################          ###################################################################
#Preliminaries
δ = 0.95      #Discount factor
I = 0         #Continuation criterion for the following while loop


#Algorithm
while I == 0
    ConstantState = Model(with_optimizer(Ipopt.Optimizer, max_iter = 100))
    @variable(ConstantState, 0 <= μ11 <= 1, start = rand() )
    @variable(ConstantState, 0 <= a <= 1, start = rand() )
    @variable(ConstantState, 0 <= b <= 1, start = rand() )
    @constraint(ConstantState, μ11 == 3*a - 15*b + 6)
    @constraint(ConstantState, μ11 == 17*a - b - 7)
    @constraint(ConstantState, μ11 <= a )
    @constraint(ConstantState, μ11 <= b )
    @constraint(ConstantState, 0 <= 1 - a - b + μ11 <= 1)
    @NLobjective(ConstantState, Max, 4 + μ11 - 2*(a + b))
    JuMP.optimize!(ConstantState)

    global μ11star = JuMP.value.(μ11)                       #Saving the results for validation in the second part
    global astar = JuMP.value.(a)
    global bstar = JuMP.value.(b)
    global μ12star = astar - μ11star
    global μ21star = bstar - μ11star
    global μ22star = 1 - astar - bstar + μ11star
    global Vstar = (4 + μ11star -2(astar+bstar))/(1-δ)

    ConstantState_t = Model(with_optimizer(Ipopt.Optimizer, max_iter = 100))
    @variable(ConstantState_t, 0 <= μ11_t <= 1, start = rand() )
    @constraint(ConstantState_t, μ11_t <= astar )
    @constraint(ConstantState_t, μ11_t <= bstar )
    @constraint(ConstantState_t, 0 <= 1 - astar - bstar + μ11_t <= 1)
    @NLobjective(ConstantState_t, Max, 4 + μ11_t - 2*(astar + bstar) + δ*Vstar)
    JuMP.optimize!(ConstantState_t)                         #Validating the results obtained above

    global μ11star_t = JuMP.value.(μ11_t)
    global μ12star_t = astar - μ11star_t
    global μ21star_t = bstar - μ11star_t
    global μ22star_t = 1 - astar - bstar + μ11star_t
    global Vstar_t = (4 + μ11star_t - 2(astar+bstar))/(1-δ)
    global astar_t = 0.6*μ11star_t + 0.5*μ12star_t + 0.4*μ21star_t + 0.35*μ22star_t
    global bstar_t = 0.65*μ11star_t + 0.45*μ12star_t + 0.55*μ21star_t + 0.3*μ22star_t

    @show astar - astar_t                                   #The differences between first and second part results
    @show bstar - bstar_t
    @show Vstar - Vstar_t

    if abs(astar_t - astar) > 10^(-7) || abs(bstar_t - bstar) > 10^(-7) || abs(Vstar_t - Vstar) > 10^(-7)
        global I = 0
    else
        global I = 1
        println("We are in the constant aggregate state that is ($astar_t, $bstar_t). The equilibrium matching in the constant aggregate state
        is [μ11 = $μ11star, μ12 = $μ12star, μ21 = $μ21star, μ22 = $μ22star] and Vstar is $Vstar")
    end
end


###################################################################          ###################################################################
#                                                                  Question 4                                                                  #
###################################################################          ###################################################################
include("Base-HK.jl")

#Choose the discount factor, the number of basis functions per dimension, and that of nodes
δ = 0.95                                         #Discount factor
M = 6                                            #Due to function registration in JuMP, M can only be 4, 6 or 8
w = 12                                            #Number of nodes must be bigger than or equal to M+1

#Setup for the algorithm
InitialNodes = F_InitialNodes(w)                 #Generating the Chebyshev interpolation nodes
Nodes = F_Nodes(w)                               #Generating 2-dimensional nodes via cartesian product

Regressors = F_Regressors(w)                     #Calculating the regressor values
SquaredRegressors = F_SquaredRegressors(w)       #Calculating the squared regressors only once
ystars = []                                      #To store the values of objective function at optimum for each state

InitialNodesAdjusted = F_InitialNodesAdjusted(w) #Adjusting the Chebyshev interpolation nodes
NodesAdjusted = F_NodesAdjusted(w)               #Cartesian product of nodes for two dimensional approximation

μstars = []                                      #To store optimal policies at each state
α = ones(M,M)                                    #Starting values for the Chebyshev coefficients

I = 0                                            #Auxiliary object to control the while loop
reps = 0                                         #Auxiliary object to keep track of the number of iterations
Tolerance = 10^(-4)
################################################################### Algorithm ###################################################################
@time begin
    if M ==4            #The objective function is constructed only for the case where M = 4
    while I == 0        #Keeps the algorithm running according to the the stopping criterion

        global Tolerance
        global reps
        reps += 1       #Keeps track of the number of iterations

        global NodesAdjusted
        global α

        ystar = zeros(w,w)                             #Temporarily storing the value of objective function at each iteration
        μstar = [zeros(2,2) for j in 1:w, j in 1:w]    #Temporarily storing the optimal policies at a given state

        for j1 in 1:w
            for j2 in 1:w
                VFA = Model(with_optimizer(Ipopt.Optimizer, max_iter = 1000))
                @variable(VFA, 0 <= μ[1:2, 1:2] <= 1, start = rand())
                @constraint(VFA, μ[1,2] + μ[1,1] == NodesAdjusted[j1,j2][1])
                @constraint(VFA, μ[2,1] + μ[1,1] == NodesAdjusted[j1,j2][2])
                @constraint(VFA, μ[2,2] - μ[1,1] == 1 - NodesAdjusted[j1,j2][1] - NodesAdjusted[j1,j2][2])
                @NLexpression(VFA, ϕa1, 1)
                @NLexpression(VFA, ϕa2, 2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)
                @NLexpression(VFA, ϕa3, 2*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕa4, 4*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^3 - 3*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕb1, 1)
                @NLexpression(VFA, ϕb2, 2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)
                @NLexpression(VFA, ϕb3, 2*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕb4, 4*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^3 - 3*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLobjective(VFA, Max, μ[1,1] + 2*μ[2,1] + 2*μ[1,2] + 4*μ[2,2] + δ*(ϕa1 * (α[1,1]*ϕb1 + α[1,2]*ϕb2 + α[1,3]*ϕb3 + α[1,4]*ϕb4)
                                                                                    +ϕa2 * (α[2,1]*ϕb1 + α[2,2]*ϕb2 + α[2,3]*ϕb3 + α[2,4]*ϕb4)
                                                                                    +ϕa3 * (α[3,1]*ϕb1 + α[3,2]*ϕb2 + α[3,3]*ϕb3 + α[3,4]*ϕb4)
                                                                                    +ϕa4 * (α[4,1]*ϕb1 + α[4,2]*ϕb2 + α[4,3]*ϕb3 + α[4,4]*ϕb4)
                                                                                    )
                            )
                JuMP.optimize!(VFA)

                ystar[j1,j2] = objective_value(VFA)
                μstar[j1,j2][1,1] = JuMP.value.(μ[1,1])
                μstar[j1,j2][1,2] = JuMP.value.(μ[1,2])
                μstar[j1,j2][2,1] = JuMP.value.(μ[2,1])
                μstar[j1,j2][2,2] = JuMP.value.(μ[2,2])
            end
        end

        for m1 in 1:M           #Updating the Chebyshev coefficients
            for m2 in 1:M
                global α[m1,m2] = (sum(ystar[j1,j2]*Regressors[j1,j2][m1,m2] for j1 in 1:w, j2 in 1:w))/SquaredRegressors[m1,m2]
            end
        end

        global ystars           #Keeping track of the values of objective function at optimum for each iteration
        ystars = push!(ystars,ystar)

        global μstars           #Keeping track of the suboptimal policies at each iteration
        μstars = push!(μstars,μstar)

        if reps != 1
            global Difference = sum(abs, ystars[reps] - ystars[reps-1])    #Bellman equation errors according to the last two approximation
            if Difference < Tolerance                                      #The stopping criterion
                println("Bellman equation errors for $w*$w interpolation nodes with $M basis functions per dimension
                        stopped decreasing after $reps iterations. The latest difference is $Difference.")
                CSV.write(joinpath(@__DIR__,"ystar_M=4_w=$w.csv"), DataFrame(ystar))
                CSV.write(joinpath(@__DIR__,"mustar_M=4_w=$w.csv"), DataFrame(μstar))
                CSV.write(joinpath(@__DIR__,"alpha_M=4_w=$w.csv"), DataFrame(α))
                global I = 1
            else
                global I = 0
            end
        else
            global I = 0
        end
    end
    elseif M == 6              #Follows the exact same procuder above for the case where M = 6
    while I == 0

        global Tolerance
        global reps
        reps += 1

        global NodesAdjusted
        global α

        ystar = zeros(w,w)
        μstar = [zeros(2,2) for j in 1:w, j in 1:w]
        NodesAdjustedPrime = [zeros(1,2) for j in 1:w, j in 1:w]

        for j1 in 1:w
            for j2 in 1:w
                VFA = Model(with_optimizer(Ipopt.Optimizer, max_iter = 1000))
                @variable(VFA, 0 <= μ[1:2, 1:2] <= 1, start = rand())
                @constraint(VFA, μ[1,2] + μ[1,1] == NodesAdjusted[j1,j2][1])
                @constraint(VFA, μ[2,1] + μ[1,1] == NodesAdjusted[j1,j2][2])
                @constraint(VFA, μ[2,2] - μ[1,1] == 1 - NodesAdjusted[j1,j2][1] - NodesAdjusted[j1,j2][2])
                @NLexpression(VFA, ϕa1, 1)
                @NLexpression(VFA, ϕa2, (2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕa3, 2*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕa4, 4*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^3 - 3*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕa5, 8*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^4 - 8*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^2 + 1)
                @NLexpression(VFA, ϕa6, 16*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^5 - 20*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^3 + 5*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕb1, 1)
                @NLexpression(VFA, ϕb2, (2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLexpression(VFA, ϕb3, 2*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕb4, 4*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^3 - 3*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLexpression(VFA, ϕb5, 8*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^4 - 8*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^2 + 1)
                @NLexpression(VFA, ϕb6, 16*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^5 - 20*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^3 + 5*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLobjective(VFA, Max, μ[1,1] + 2μ[2,1] + 2μ[1,2] + 4μ[2,2] + δ*(ϕa1 * (α[1,1]*ϕb1 + α[1,2]*ϕb2 + α[1,3]*ϕb3 + α[1,4]*ϕb4+ α[1,5]*ϕb5+ α[1,6]*ϕb6)
                                                                                +ϕa2 * (α[2,1]*ϕb1 + α[2,2]*ϕb2 + α[2,3]*ϕb3 + α[2,4]*ϕb4+ α[2,5]*ϕb5+ α[2,6]*ϕb6)
                                                                                +ϕa3 * (α[3,1]*ϕb1 + α[3,2]*ϕb2 + α[3,3]*ϕb3 + α[3,4]*ϕb4+ α[3,5]*ϕb5+ α[3,6]*ϕb6)
                                                                                +ϕa4 * (α[4,1]*ϕb1 + α[4,2]*ϕb2 + α[4,3]*ϕb3 + α[4,4]*ϕb4+ α[4,5]*ϕb5+ α[4,6]*ϕb6)
                                                                                +ϕa5 * (α[5,1]*ϕb1 + α[5,2]*ϕb2 + α[5,3]*ϕb3 + α[5,4]*ϕb4+ α[5,5]*ϕb5+ α[5,6]*ϕb6)
                                                                                +ϕa6 * (α[6,1]*ϕb1 + α[6,2]*ϕb2 + α[6,3]*ϕb3 + α[6,4]*ϕb4+ α[6,5]*ϕb5+ α[6,6]*ϕb6)
                                                                             )
                        )
                JuMP.optimize!(VFA)

                ystar[j1,j2] = objective_value(VFA)
                μstar[j1,j2][1,1] = JuMP.value.(μ[1,1])
                μstar[j1,j2][1,2] = JuMP.value.(μ[1,2])
                μstar[j1,j2][2,1] = JuMP.value.(μ[2,1])
                μstar[j1,j2][2,2] = JuMP.value.(μ[2,2])
            end
        end

        for m1 in 1:M
            for m2 in 1:M
                global α[m1,m2] = (sum(ystar[j1,j2]*Regressors[j1,j2][m1,m2] for j1 in 1:w, j2 in 1:w))/SquaredRegressors[m1,m2]
            end
        end

        global ystars
        ystars = push!(ystars,ystar)

        global μstars
        μstars = push!(μstars,μstar)

        if reps != 1
            global Difference = sum(abs, ystars[reps] - ystars[reps-1])
            if Difference < Tolerance
                println("Bellman equation errors for $w*$w interpolation nodes with $M basis functions per dimension
                        stopped decreasing after $reps iterations. The latest difference is $Difference.")
                CSV.write(joinpath(@__DIR__,"ystar_M=6_w=$w.csv"), DataFrame(ystar))
                CSV.write(joinpath(@__DIR__,"mustar_M=6_w=$w.csv"), DataFrame(μstar))
                CSV.write(joinpath(@__DIR__,"alpha_M=6_w=$w.csv"), DataFrame(α))
                global I = 1
            else
                global I = 0
            end
        else
            global I = 0
        end
    end
    else                   #Follows the exact same procuder above for the case where M = 8
    while I == 0

        global Tolerance
        global reps
        reps += 1

        global NodesAdjusted
        global α

        ystar = zeros(w,w)
        μstar = [zeros(2,2) for j in 1:w, j in 1:w]
        NodesAdjustedPrime = [zeros(1,2) for j in 1:w, j in 1:w]

        for j1 in 1:w
            for j2 in 1:w
                VFA = Model(with_optimizer(Ipopt.Optimizer, max_iter = 1000))
                @variable(VFA, 0 <= μ[1:2, 1:2] <= 1, start = rand())
                @constraint(VFA, μ[1,2] + μ[1,1] == NodesAdjusted[j1,j2][1])
                @constraint(VFA, μ[2,1] + μ[1,1] == NodesAdjusted[j1,j2][2])
                @constraint(VFA, μ[2,2] - μ[1,1] == 1 - NodesAdjusted[j1,j2][1] - NodesAdjusted[j1,j2][2])
                @NLexpression(VFA, ϕa1, 1)
                @NLexpression(VFA, ϕa2, (2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕa3, 2*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕa4, 4*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^3 - 3*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕa5, 8*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^4 - 8*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^2 + 1)
                @NLexpression(VFA, ϕa6, 16*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^5 - 20*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^3 + 5*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕa7, 32*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^6 - 48*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^4 + 18*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕa8, 64*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^7 - 112*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^5 + 56*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1)^3 - 7*(2*(0.60μ[1,1] + 0.40μ[2,1] + 0.50μ[1,2] + 0.35μ[2,2]) - 1))
                @NLexpression(VFA, ϕb1, 1)
                @NLexpression(VFA, ϕb2, (2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLexpression(VFA, ϕb3, 2*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕb4, 4*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^3 - 3*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLexpression(VFA, ϕb5, 8*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^4 - 8*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^2 + 1)
                @NLexpression(VFA, ϕb6, 16*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^5 - 20*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^3 + 5*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLexpression(VFA, ϕb7, 32*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^6 - 48*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^4 + 18*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^2 - 1)
                @NLexpression(VFA, ϕb8, 64*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^7 - 112*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^5 + 56*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1)^3 - 7*(2*(0.65μ[1,1] + 0.55μ[2,1] + 0.45μ[1,2] + 0.30μ[2,2]) - 1))
                @NLobjective(VFA, Max, μ[1,1] + 2μ[2,1] + 2μ[1,2] + 4μ[2,2] + δ*(ϕa1 * (α[1,1]*ϕb1 + α[1,2]*ϕb2 + α[1,3]*ϕb3 + α[1,4]*ϕb4+ α[1,5]*ϕb5+ α[1,6]*ϕb6+ α[1,7]*ϕb7+ α[1,8]*ϕb8)
                                                                                        +ϕa2 * (α[2,1]*ϕb1 + α[2,2]*ϕb2 + α[2,3]*ϕb3 + α[2,4]*ϕb4+ α[2,5]*ϕb5+ α[2,6]*ϕb6+ α[2,7]*ϕb7+ α[2,8]*ϕb8)
                                                                                        +ϕa3 * (α[3,1]*ϕb1 + α[3,2]*ϕb2 + α[3,3]*ϕb3 + α[3,4]*ϕb4+ α[3,5]*ϕb5+ α[3,6]*ϕb6+ α[3,7]*ϕb7+ α[3,8]*ϕb8)
                                                                                        +ϕa4 * (α[4,1]*ϕb1 + α[4,2]*ϕb2 + α[4,3]*ϕb3 + α[4,4]*ϕb4+ α[4,5]*ϕb5+ α[4,6]*ϕb6+ α[4,7]*ϕb7+ α[4,8]*ϕb8)
                                                                                        +ϕa5 * (α[5,1]*ϕb1 + α[5,2]*ϕb2 + α[5,3]*ϕb3 + α[5,4]*ϕb4+ α[5,5]*ϕb5+ α[5,6]*ϕb6+ α[5,7]*ϕb7+ α[5,8]*ϕb8)
                                                                                        +ϕa6 * (α[6,1]*ϕb1 + α[6,2]*ϕb2 + α[6,3]*ϕb3 + α[6,4]*ϕb4+ α[6,5]*ϕb5+ α[6,6]*ϕb6+ α[6,7]*ϕb7+ α[6,8]*ϕb8)
                                                                                        +ϕa7 * (α[7,1]*ϕb1 + α[7,2]*ϕb2 + α[7,3]*ϕb3 + α[7,4]*ϕb4+ α[7,5]*ϕb5+ α[7,6]*ϕb6+ α[7,7]*ϕb7+ α[7,8]*ϕb8)
                                                                                        +ϕa8 * (α[8,1]*ϕb1 + α[8,2]*ϕb2 + α[8,3]*ϕb3 + α[8,4]*ϕb4+ α[8,5]*ϕb5+ α[8,6]*ϕb6+ α[8,7]*ϕb7+ α[8,8]*ϕb8)
                                                                                        )
                            )
                JuMP.optimize!(VFA)

                ystar[j1,j2] = objective_value(VFA)
                μstar[j1,j2][1,1] = JuMP.value.(μ[1,1])
                μstar[j1,j2][1,2] = JuMP.value.(μ[1,2])
                μstar[j1,j2][2,1] = JuMP.value.(μ[2,1])
                μstar[j1,j2][2,2] = JuMP.value.(μ[2,2])
            end
        end

        for m1 in 1:M
            for m2 in 1:M
                global α[m1,m2] = (sum(ystar[j1,j2]*Regressors[j1,j2][m1,m2] for j1 in 1:w, j2 in 1:w))/SquaredRegressors[m1,m2]
            end
        end

        global ystars
        ystars = push!(ystars,ystar)

        global μstars
        μstars = push!(μstars,μstar)

        if reps != 1
            global Difference = sum(abs, ystars[reps] - ystars[reps-1])
            if Difference < Tolerance
                println("Bellman equation errors for $w*$w interpolation nodes with $M basis functions per dimension
                        stopped decreasing after $reps iterations. The latest difference is $Difference.")
                CSV.write(joinpath(@__DIR__,"ystar_M=8_w=$w.csv"), DataFrame(ystar))
                CSV.write(joinpath(@__DIR__,"mustar_M=8_w=$w.csv"), DataFrame(μstar))
                CSV.write(joinpath(@__DIR__,"alpha_M=8_w=$w.csv"), DataFrame(α))
                global I = 1
            else
                global I = 0
            end
        else
            global I = 0
        end
    end
    end
end


#Writing NodesAdjusted for the policy function analysis
CSV.write(joinpath(@__DIR__,"NodesAdjusted.csv"), DataFrame(NodesAdjusted))
