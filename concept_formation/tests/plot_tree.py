from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize

if __name__ == "__main__":

    tree = MultinomialCobwebTree(3, # 1=CU, 2=MI, 3=NMI
                             1.0, # alpha weight
                             True, # dynamically compute alpha
                             True, # weight attr by avg occurance of attr
                             True, # categorize to basic level (true)? or leaf (false)?
                             False) # predict using mixture at root (true)? or single best (false)?

    with open('cobweb_w_freq_5_rocstories_tree.json', 'r') as fin:
        tree_data = fin.readlines()

    tree_data = "".join(tree_data)
    tree.load_json(tree_data)
    visualize(tree)
