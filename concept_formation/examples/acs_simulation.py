from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from concept_formation.evaluation import incremental_evaluation
from concept_formation.trestle import TrestleTree
from concept_formation.datasets import load_rb_s_07
from concept_formation.preprocessor import ObjectVariablizer

def convert_cat(tower):
    new_tower = {}
    for attr in tower:
        if isinstance(tower[attr], dict):
            new_b = {}
            for inner in tower[attr]:
                if inner == 'type':
                    new_b[inner] = tower[attr][inner]
                else:
                    new_b[inner] = str(round(tower[attr][inner] * 2.0) / 2.0)
            new_tower[attr] = new_b
        else:
            new_tower[attr] = tower[attr]
    return new_tower

if __name__ == "__main__":
    num_runs = 1000
    num_examples = 30
    towers = load_rb_s_07()

    variablizer = ObjectVariablizer()
    towers = [convert_cat(variablizer.transform(t)) for t in towers]

    trestle_data = incremental_evaluation(TrestleTree(), towers,
                                          run_length=num_examples, runs=num_runs,
                                          attr="success")

    with open('acs_trestle_data.csv', "w") as fout:
        fout.write("opportunity,accuracy\n")
        for point in trestle_data:
            fout.write("%s,%s\n" % (point[0], point[1]))

