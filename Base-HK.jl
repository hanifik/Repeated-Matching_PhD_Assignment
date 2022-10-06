using LinearAlgebra, Random

#=
Since there is no way to implement the The Chebyshev polynomials of the first kind in Julia, to the best of my knowledge,
the following function helps storing the first few Chebyshev polynomials of the first kind
=#

function ChebyshevT(n::Int64, x::Float64)
    if n == 1
        return 1
    elseif n == 2
        return x
    else
        T = zeros(n)
        T[1] = 1
        T[2] = x
        for k in 3:n
            T[k] = 2*x*T[k-1] - T[k-2]
        end
        return T[n]
    end
end


function F_InitialNodes(w::Int64)
    InitialNodes = zeros(w)
    for j in 1:w
        InitialNodes[j] = -cos((2j-1)/(2*w)*π)
    end
    return InitialNodes
end


function F_Nodes(w::Int64)
    Nodes = [zeros(1,2) for j1 in 1:w, j2 in 1:w]
    for j1 in 1:w
        for j2 in 1:w
            Nodes[j1,j2]=[InitialNodes[j1] InitialNodes[j2]]
        end
    end
    return Nodes
end


function F_Regressors(w::Int64)
    Regressors = [zeros(M,M) for j1 in 1:w, j2 in 1:w]
    for j1 in 1:w
        for j2 in 1:w
            for m1 in 1:M
                for m2 in 1:M
                    Regressors[j1,j2][m1,m2] = ChebyshevT(m1,Nodes[j1,j2][1])*ChebyshevT(m2,Nodes[j1,j2][2])
                end
            end
        end
    end
    return Regressors
end


function F_SquaredRegressors(w::Int64)
    SquaredRegressors = zeros(M,M)
    for m1 in 1:M
        for m2 in 1:M
            SquaredRegressors[m1,m2] = sum(  (ChebyshevT(m1,InitialNodes[j1])^2)*(ChebyshevT(m2,InitialNodes[j2])^2) for j1 in 1:w, j2 in 1:w )
        end
    end
    return SquaredRegressors
end


function F_InitialNodesAdjusted(w::Int64)
    InitialNodesAdjusted = zeros(w)
    for j in 1:w
        InitialNodesAdjusted[j] = (InitialNodes[j] + 1)/2
    end
    return InitialNodesAdjusted
end


function F_NodesAdjusted(w::Int64)
    NodesAdjusted = [zeros(1,2) for j1 in 1:w, j2 in 1:w]
    for j1 in 1:w
        for j2 in 1:w
            NodesAdjusted[j1,j2]=[InitialNodesAdjusted[j1] InitialNodesAdjusted[j2]]
        end
    end
    return NodesAdjusted
end
