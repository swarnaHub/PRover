def get_proofs(f):
    proofs = []
    lines = f.read().splitlines()
    for line in lines:
        if line == "[]":
            proofs.append([])
            continue
        curr_proof = []
        line = line[2:-2].split("), (")
        for edge in line:
            edge = edge.split(", ")
            curr_proof.append((edge[0][1:-1], edge[1][1:-1]))

        proofs.append(curr_proof)

    return proofs

if __name__ == '__main__':
    f1 = open("../data/old_proof.txt", "r", encoding="utf-8-sig")
    f2 = open("../data/new_proof.txt", "r", encoding="utf-8-sig")

    old_proofs = get_proofs(f1)
    new_proofs = get_proofs(f2)

    for i in range(len(old_proofs)):
        if set(old_proofs[i]) != set(new_proofs[i]):
            print(i)