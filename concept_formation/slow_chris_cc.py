from concept_formation.cobweb import CobwebNode
from time import time

from tqdm import tqdm

from concept_formation.cobweb import CobwebTree
from visualize import visualize
from train_contextual_cobweb import _load_text

def get_path(node):
    path = [node]
    while path[-1].parent:
        path.append(path[-1].parent)
    return path

class ContextualCobwebTree(CobwebTree):

    def __init__(self, window):
        """
        Note window only specifies how much context to add to each side,
        doesn't include the anchor word.

        E.g., to get a window with 2 before and 2 words after the anchor, then
        set the window=2
        """
        super().__init__()
        self.root = ContextualCobwebNode()
        self.root.tree = self

        self.window = window
    
    def fit_to_text(self, text):
        context_nodes = []

        for anchor_idx in tqdm(range(len(text))):
            while ((len(context_nodes) < anchor_idx + self.window + 1) and
                   len(context_nodes) < len(text)):
                context_nodes.append(self.categorize({'anchor': text[len(context_nodes)]}))

            for _ in range(3):
                for i in range(2*self.window + 1):
                    idx = anchor_idx - self.window + i
                    if idx < 0 or idx >= len(text): 
                        continue
                    instance = self.create_instance(idx, self.window, text, context_nodes)
                    context_nodes[idx] = self.categorize(instance)

            instance = self.create_instance(anchor_idx, self.window, text, context_nodes)
            instance['_idx'] = str(anchor_idx)

            context_nodes[anchor_idx] = self.ifit(instance)

    def create_instance(self, anchor_idx, window, text, context_nodes):
        context = context_nodes[max(0, anchor_idx-self.window): anchor_idx]
        context += context_nodes[anchor_idx+1:anchor_idx+self.window+1]

        instance = {}
        for n in context:
            instance[n] = True

        instance['anchor'] = text[anchor_idx]
        return instance

class ContextualCobwebNode(CobwebNode):

    def __init__(self, otherNode=None):
        super().__init__(otherNode)

    def get_concepts(self):
        s = set([self.concept_id])
        for c in self.children:
            s.update(c.get_concepts())
        return s

    def test_valid(self, valid_concepts):
        for attr in self.av_counts:
            if isinstance(attr, ContextualCobwebNode):
                assert attr.concept_id in valid_concepts
        for c in self.children:
            c.test_valid(valid_concepts)

    def __hash__(self):
        return hash(self.concept_id)

    def __str__(self):
        return "Concept-{}".format(self.concept_id)

    def __getitem__(self, indices):
        return True

    def expected_correct_guesses(self):
        """
        Modified to handle attribute that are concepts to tally and compute
        correct guesses over all concepts in path.
        """
        correct_guesses = 0.0
        attr_count = 0
        concept_attr_count = 0
        concept_vals = {}

        for attr in self.attrs():
            attr_count += 1

            if isinstance(attr, ContextualCobwebNode):
                concept_attr_count += self.av_counts[attr][True]

                curr = attr
                concept_vals[attr] = (concept_vals.get(attr, 0) +
                        self.av_counts[attr][True])
                while curr.parent:
                    curr = curr.parent
                    concept_vals[curr] = (concept_vals.get(curr, 0) +
                            self.av_counts[attr][True])

            else:
                for val in self.av_counts[attr]:
                    prob = (self.av_counts[attr][val]) / self.count
                    correct_guesses += (prob * prob)

        for c in concept_vals:
            prob = concept_vals[c] / concept_attr_count
            correct_guesses += (prob * prob)
            correct_guesses += ((1-prob) * (1-prob))

        return correct_guesses / attr_count
    
    def output_json(self):
        """
        Outputs the categorization tree in JSON form.
        This is a modification of the :meth:`CobwebNode.output_json
        <concept_formation.cobweb.CobwebNode.output_json>` to handle numeric
        values.
        :return: an object that contains all of the structural information of
            the node and its children
        :rtype: obj
        """
        output = {}
        output["name"] = "Concept" + str(self.concept_id)
        output["size"] = self.count
        output["children"] = []

        temp = {}

        concept_attr_count = 0
        concept_vals = {}

        for attr in self.attrs('all'):

            if isinstance(attr, ContextualCobwebNode):
                assert True in self.av_counts[attr]
                assert len(self.av_counts[attr]) == 1
                concept_attr_count += self.av_counts[attr][True]

                curr = attr
                concept_vals[curr] = (concept_vals.get(curr, 0) +
                        self.av_counts[attr][True])

                while curr.parent:
                    curr = curr.parent
                    concept_vals[curr] = (concept_vals.get(curr, 0) +
                            self.av_counts[attr][True])

            else:
                temp[str(attr)] = {}
                for val in self.av_counts[attr]:
                    temp[str(attr)][str(val)] = self.av_counts[attr][val]

        for c in concept_vals:
            temp[str(c)] = {True: concept_vals[c] * self.count / concept_attr_count}

        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp

        return output


if __name__ == "__main__":

    tree = ContextualCobwebTree(window=2)

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
                stop_words][:300]

        print('iterations needed', len(text))
        start = time()
        tree.fit_to_text(text)
        print(time()-start)
        print(text_num)
    visualize(tree)

    valid_concepts = tree.root.get_concepts()
    tree.root.test_valid(valid_concepts)







