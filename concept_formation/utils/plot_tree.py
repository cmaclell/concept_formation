import argparse

from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot a MultinomialCobwebTree from a json checkpoint.')
    parser.add_argument('tree_checkpoint', type=open, help="A json checkpoint file")
    args = parser.parse_args()

    tree = MultinomialCobwebTree()
    tree_data = args.tree_checkpoint.read()
    tree.load_json(tree_data)
    visualize(tree)

