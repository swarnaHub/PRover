class Node:
    def __init__(self, head):
        self.head = head

    def __str__(self):
        return str(self.head)

def get_proof_graph(proof_str):
    stack = []
    last_open = 0
    last_open_index = 0
    pop_list = []
    all_edges = []
    all_nodes = []

    proof_str = proof_str.replace("(", " ( ")
    proof_str = proof_str.replace(")", " ) ")
    proof_str = proof_str.split()

    should_join = False
    for i in range(len(proof_str)):

        _s = proof_str[i]
        x = _s.strip()
        if len(x) == 0:
            continue

        if x == "(":
            stack.append((x, i))
            last_open = len(stack) - 1
            last_open_index = i
        elif x == ")":
            for j in range(last_open + 1, len(stack)):
                if isinstance(stack[j][0], Node):
                    pop_list.append((stack[j][1], stack[j][0]))

            stack = stack[:last_open]
            for j in range((len(stack))):
                if stack[j][0] == "(":
                    last_open = j
                    last_open_index = stack[j][1]

        elif x == '[' or x == ']':
            pass
        elif x == "->":
            should_join = True
        else:
            # terminal
            if x not in all_nodes:
                all_nodes.append(x)

            if should_join:
                # A new terminal node can get appended to a maximum of two nodes
                #assert(len(pop_list)) <= 4

                new_pop_list = []
                for (index, p) in pop_list:
                    if index < last_open_index:
                        new_pop_list.append((index, p))
                    else:
                        all_edges.append((p.head, x))
                pop_list = new_pop_list

            stack.append((Node(x), i))

            should_join = False

        print(last_open_index)
        print(stack)
        print(pop_list)
        print("\n")

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

if __name__ == '__main__':
    all_nodes, all_edges = get_proof_graph("[(((((((triple1) -> rule8) ((((triple1) -> rule8) triple1) -> rule7)) -> rule5) ((((triple1) -> rule8) ((((triple1) -> rule8) ((((triple1) -> rule8) triple1) -> rule7)) -> rule5)) -> rule4)) -> rule3)")
    #all_nodes, all_edges = get_proof_graph("[(((((triple5) -> rule7) ((((((((triple5) -> rule7)) -> rule2) triple5) -> rule6) triple5) -> rule5)) -> rule1))]")
    print(all_edges)