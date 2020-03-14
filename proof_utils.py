class Node:
    def __init__(self, head, elements):
        self.head = head
        self.elements = elements

    def __str__(self):
        return str(self.head)

def get_proof_graph(proof_str):
    stack = []
    last_open = 0
    pop_list = []
    all_edges = []
    all_nodes = []

    proof_str = proof_str.replace("(", " ( ")
    proof_str = proof_str.replace(")", " ) ")
    proof_str = proof_str.split()

    for i in range(len(proof_str)):
        _s = proof_str[i]
        x = _s.strip()
        if len(x) == 0:
            continue

        if x == "(":
            stack.append(x)
            last_open = len(stack) - 1
        elif x == ")":
            for j in range(last_open + 1, len(stack)):
                if isinstance(stack[j], Node):
                    pop_list.append(stack[j])

            stack = stack[:last_open]
            for j in range((len(stack))):
                if stack[j] == "(":
                    last_open = j
            for p in pop_list:
                stack.append(p)
            pop_list = []
        elif x == "->" or x == '[' or x == ']':
            pass
        else:
            # terminal
            if x not in all_nodes:
                all_nodes.append(x)
            for j in range(last_open + 1, len(stack)):
                if isinstance(stack[j], Node):
                    pop_list.append(stack[j])

            stack = stack[:last_open + 1]

            if len(pop_list) == 0:
                stack.append(Node(x, [x]))
            else:
                all_elems = []
                # A new terminal node can get appended to a maximum of two nodes
                assert(len(pop_list)) <= 2

                for p in pop_list:
                    all_edges.append((p.head, x))
                    for y in p.elements:
                        all_elems.append(y)
                pop_list = []

                stack.append(Node(x, all_elems))

    return all_nodes, all_edges

def get_proof_graph_with_fail(proof_str):
    proof_str = proof_str[:-2].split("=")[1].strip()[1:-1]
    nodes = proof_str.split(" <- ")

    all_nodes = []
    all_edges = []
    for i in range(len(nodes)-1):
        all_nodes.append(nodes[i])
        if nodes[i+1] != "FAIL":
            all_edges.append((nodes[i+1], nodes[i]))

    return all_nodes, all_edges


