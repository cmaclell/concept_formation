import itertools
from random import normalvariate
from random import choice
from random import random
from cobweb3 import Cobweb3Tree

class Labyrinth(Cobweb3Tree):

    def _labyrinth(self, instance):
        """
        Recursively calls labyrinth on all of the components in a depth-first
        traversal. Once all of the components have been classified then then it
        classifies the current node.
        """
        temp_instance = {}

        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self._labyrinth(instance[attr])
            elif isinstance(instance[attr], list):
                temp_instance[tuple(instance[attr])] = True
            else:
                temp_instance[attr] = instance[attr]

        # should be able to match just at the root, if the matchings change
        # than the counts between parent and child will be thrown off which is
        # not allowed to happen so for now don't worry about it.
        # TODO check if this needs to be changed
        temp_instance = self._match(temp_instance)
        ret = self._cobweb(temp_instance)

        return ret

    def _rename(self, instance, mapping):
        """
        Given a mapping (type = dict) rename the 
        components and relations and return the renamed
        instance.
        """
        # Ensure it is a complete mapping
        # Might be troublesome if there is a name collision
        for attr in instance:
            if not isinstance(instance[attr], Labyrinth):
                continue
            if attr not in mapping:
                mapping[attr] = attr

        temp_instance = {}
        relations = []

        # rename all attribute values
        for attr in instance:
            if isinstance(attr, tuple):
                relations.append(attr)
            elif isinstance(instance[attr], Labyrinth):
                mapping[attr]
                instance[attr]
                temp_instance[mapping[attr]] = instance[attr]
            else:
                temp_instance[attr] = instance[attr]

        #rename relations and add them to instance
        for relation in relations:
            temp = []
            for idx, val in enumerate(relation):
                if idx == 0:
                    temp.append(val)
                else:
                    temp.append(mapping[val])
            temp_instance[tuple(temp)] = True

        return temp_instance

    def _exhaustive_match(self, instance):
        """
        Compute matches exhaustively. This has O(n!) time complexity, so it
        won't scale very well.
        """
        from_name = [attr for attr in instance if isinstance(instance[attr],
                                                             Labyrinth)]
        to_name = []
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Labyrinth):
                    to_name.append(attr)
                    break

        mappings = []
        if len(from_name) > len(to_name):
            from_lists = set([x[:len(to_name)] for x in
                             itertools.permutations(from_name)])
            for from_list in from_lists:
                mapping = {}
                for i in range(len(to_name)):
                    mapping[from_list[i]] = to_name[i]
                mappings.append(mapping)
        else:
            to_lists = set([x[:len(from_name)] for x in 
                            itertools.permutations(to_name)])
            for to_list in to_lists:
                mapping = {}
                for i in range(len(from_name)):
                    mapping[from_name[i]] = to_list[i]
                mappings.append(mapping)
    
        scored_mappings = []
        for mapping in mappings:
            temp = self._shallow_copy()
            temp._increment_counts(self._rename(instance, mapping))
            score = temp._expected_correct_guesses()
            scored_mappings.append((score, mapping))

        best_mapping = sorted(scored_mappings, key=lambda x: x[0])[0][1]
        return best_mapping

    def _match(self, instance):
        """ 
        Define the specialized matching function to rename components
        and relations to maximize the match between instance and the
        current concept (self).
        """
        mapping = self._exhaustive_match(instance)
        temp_instance = self._rename(instance, mapping)
        return temp_instance

    #def _cobweb(self, instance):
    #    super(Labyrinth, self)._cobweb(instance)

    def _labyrinth_categorize(self, instance):
        temp_instance = {}

        for attr in instance:
            for val in instance[attr]:
                if isinstance(val, dict):
                    temp_instance[attr] = self._labyrinth_categorize(val)
                else:
                    temp_instance[attr] = val

        return self._cobweb_categorize(temp_instance)

    def ifit(self, instance):
        self._labyrinth(instance)

    def _pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        tabs = "\t" * depth
        ret = str(('\t' * depth) + "|-" + "[" + self.concept_name + "]: " +
                  str(self.count) + "\n" + tabs + "  ")

        attributes = []

        for attr in self.av_counts:
            float_values = []
            values = []

            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    float_values.append(val)
                elif isinstance(val, Labyrinth):
                    values.append("'" + val.concept_name + "': " + 
                                  str(self.av_counts[attr][val]))
                else:
                    values.append("'" + str(val) + "': " +
                                  str(self.av_counts[attr][val]))

            if float_values:
                values.append("'mean':" + str(self._mean(float_values)))
                values.append("'std':" + str(self._std(float_values)))

            attributes.append("'" + str(attr) + "': {" + ", ".join(values) + "}")
                  
        ret += "{" + (",\n" + tabs + "   ").join(attributes) + "}\n"
        
        for c in self.children:
            ret += c._pretty_print(depth+1)

        return ret

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree.
        """
        prediction = {}

        # make a copy of the instance
        # call recursively on structured parts
        for attr in instance:
            if isinstance(instance[attr], dict):
                prediction[attr] = self.predict(instance[attr])
            else:
                prediction[attr] = instance[attr]

        concept = self._labyrinth_categorize(prediction)
        
        for attr in concept.av_counts:
            if attr in prediction:
                continue
            
            nominal_values = []
            component_values = []
            float_values = []

            for val in concept.av_counts[attr]:
                if isinstance(val, float):
                    float_values += [val] * concept.av_counts[attr][val]
                elif isinstance(val, Labyrinth):
                    component_values += [val] * concept.av_counts[attr][val] 
                else:
                    nominal_values += [val] * concept.av_counts[attr][val]

            rand = random()

            if rand < ((len(nominal_values) * 1.0) / (len(nominal_values) +
                                                      len(component_values) +
                                                      len(float_values))):
                prediction[attr] = choice(nominal_values)
            elif rand < ((len(nominal_values) + len(component_values) * 1.0) /
                         (len(nominal_values) + len(component_values) +
                          len(float_values))):
                prediction[attr] = choice(component_values).predict({})
            else:
                prediction[attr] = normalvariate(self._mean(float_values),
                                                 self._std(float_values))

        return prediction


if __name__ == "__main__":

    t = Labyrinth()
    t.train_from_json("labyrinth_test.json")
    print(t)

    test = {}
    print(t.predict(test))



