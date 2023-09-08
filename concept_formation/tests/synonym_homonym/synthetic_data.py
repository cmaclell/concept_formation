import random


grammar = {
    "SENTENCE": [
        ["S1", "V1", "O1"],
        ["S1", "V2", "O2"],
        ["S2", "V1", "O2"],
        ["S2", "V2", "O1"],
    ],
    "S1": [["a1"], ["a2"]],
    "S2": [["b1"], ["b2"]],
    "V1": [["c1"], ["c2"]],
    "V2": [["d1"], ["d2"]],
    "O1": [["e1"], ["e2"]],
    "O2": [["f1"], ["f2"]],
}


def get_nonterminals(grammar):
    return set(grammar.keys())


def create_sentence(grammar):
    out = ["SENTENCE"]
    nonterminals = get_nonterminals(grammar)

    def replace(symbols, index):
        return (
            symbols[:index]
            + random.choice(grammar[symbols[index]])
            + symbols[index + 1:]
        )

    i = 0
    while i < len(out):
        if out[i] in nonterminals:
            out = replace(out, i)
        else:
            i += 1

    return out


def get_data(num):
    return [create_sentence(grammar) for i in range(num)]


if __name__ == "__main__":
    print(get_data(10))
