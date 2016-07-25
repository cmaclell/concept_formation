import json
from pprint import pprint

from concept_formation.structure_mapper import get_component_names
from concept_formation.preprocessor import ListProcessor
from concept_formation.trestle import TrestleTree

with open("/Users/cmaclell/Downloads/trestleData.json", "r") as fin:
    data = json.load(fin)

lp = ListProcessor()
data = [lp.transform(d) for d in data]

counts = [len(get_component_names(d)) for d in data]

print("Avg number of objects in instances")
print(sum(counts) / len(counts))

t = TrestleTree()

for i,d in enumerate(data):
    print()
    print("Instance: %i" % i)
    print("Num Components: %i" % len(get_component_names(d)))
    t.ifit(d)

