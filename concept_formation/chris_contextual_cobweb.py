from concept_formation.cobweb import CobwebNode
from time import time

from tqdm import tqdm

from concept_formation.cobweb import CobwebTree
from visualize import visualize
from train_contextual_cobweb import _load_text


class ContextualCobwebTree(CobwebTree):

    def __init__(self, window):
        """
        Note window only specifies how much context to add to each side,
        doesn't include the anchor word.

        E.g., to get a window with 2 before and 2 words after the anchor, then
        set the window=2
        """
        super().__init__()
        self.window = window
    
    def fit_to_text(self, text):
        context_nodes = []

        for anchor_idx in tqdm(range(len(text))):
            while len(context_nodes) < anchor_idx + self.window:
                context_nodes.append(self.categorize(
                    {'anchor': text[len(context_nodes)]}))

            context = context_nodes[anchor_idx-self.window: anchor_idx]
            context += context_nodes[anchor_idx+1:anchor_idx+self.window+1]

            instance = {"Concept-{}".format(c.concept_id): True for c in context}
            instance['anchor'] = text[anchor_idx]

            print(instance)

            if anchor_idx > 100:
                raise Exception()

            self.ifit(instance)

if __name__ == "__main__":

    tree = ContextualCobwebTree(window=4)

    stop_words = {*"i me my myself we our ours ourselves you your yours yourself "
                   "yourselves he him his himself she her hers herself it its "
                   "itself they them their theirs themselves what which who whom "
                   "this that these those am is are was were be been being have "
                   "has had having do does did doing a an the and but if or "
                   "because as until while of at by for with about against "
                   "between into through during before after above below to from "
                   "up down in out on off over under again further then once here "
                   "there when where why how all any both each few more most "
                   "other some such no nor not only own same so than too very s t "
                   "can will just don't should now".split(' ')}

    for text_num in range(1):
        text = [word for word in _load_text(text_num) if word not in
                stop_words]# [:100]

        print('iterations needed', len(text))
        start = time()
        # run("tree.contextual_ifit(text, context_size=window_size)")
        tree.fit_to_text(text)
        print(time()-start)
        print(text_num)
    visualize(tree)







