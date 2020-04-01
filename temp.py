from pulp import *

if __name__ == '__main__':
    prob = LpProblem("NodeEdgeConsistency", LpMaximize)
    nodes = []
    for i in range(4):
        nodes.append(LpVariable("Node_" + str(i), 0, 1, LpInteger))

    edges = []
    for i in range(16):
        edges.append(LpVariable("Edge_" + str(i), 0, 1, LpInteger))

    node_logits = [0.1, 0.2, 0.3, 0.4]
    edge_logits = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]

    opt_prob = None
    for i in range(4):
        if opt_prob == None:
            opt_prob = node_logits[i] * nodes[i]
        else:
            opt_prob += node_logits[i] * nodes[i]

    for i in range(16):
        opt_prob += edge_logits[i] * edges[i]

    prob += opt_prob, "Maximum score"

    for i in range(16):
        row = int(i/4)
        col = i%4
        if row == col:
            prob += edges[i] == 0, "Self loop"
        else:
            prob += nodes[row]*nodes[col] - edges[i] >= 0, "Edge possible"

    prob.writeLP("output/NodeEdgeConsistency.lp")
    prob.solve()