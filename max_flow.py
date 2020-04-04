import numpy as np
from pulp import *

A = np.zeros((6,6))
C = np.zeros((6,6))

# assume node 0 is source and node 5 in sink. The graph in consideration has node 1,2,3,4

# symmetric A
# edge from source to node 1
A[0][1] = 1
A[1][0] = 1
C[0][1] = 4

# edges from 1,2,3,4 to sink
for i in range(1,5):
    A[i][5] = 1
    A[5][i] = 1
    C[i][5] = 1

# edges in graph
E =  [(1,2), (3,4)]
# E =  [(1,2), (3,4), (2,3)]
arcs = []   # symmetric version of E + edges from source to 1 + edges from nodes to sink
for e in E:
    arcs.append((e[0], e[1]))
    arcs.append((e[1], e[0]))
arcs.append((0,1))
for i in range(1,5):
    arcs.append((i,5))

for e in E:
    i = e[0]
    j = e[1]
    A[i][j] = 1
    A[j][i] = 1
    C[i][j] = 1000
    C[j][i] = 1000



F = []
for i in range(6):
    temp = []
    for j in range(6):
        temp.append(LpVariable("flow_var_"+str(i)+"_"+str(j), 0, 100, LpInteger ))
    F.append(temp)


prob = LpProblem("Max Flow ",LpMaximize)

prob += F[0][1], "Total Cost of Transport"
# In this formulation we say that if the graph is connected the max flow is equal to number of nodes in graph considered.
# Note, by this formulation the max flow is not zero when graph is not connected as this captures 
# only the amount we can take away from the source and put in the graph
# This is alright as the souce is connected to just one node

for n in range(1, 5):
    prob += (lpSum([F[i][j] for (i,j) in arcs if j == n]) ==
             lpSum([F[i][j] for (i,j) in arcs if i == n])), \
            "Flow Conservation in Node " + str(n)

for i in range(6):
    for j in range(6):
        prob += F[i][j] <= C[i][j] , "Capacity constraint "+str(i)+" "+str(j)


prob.writeLP("max_flow.lp")
prob.solve()
print ("Max flow = ", value(prob.objective))