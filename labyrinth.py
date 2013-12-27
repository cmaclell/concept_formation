import itertools
from cobweb3 import Cobweb3Tree

class Labyrinth(Cobweb3Tree):

    # TODO naming convention on these category utility functions is weird and
    # probably needs to be redone
    def _category_utility_component(self, parent):
        category_utility = 0.0
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Labyrinth):
                    category_utility += ((self.av_counts[attr][val] /
                                          self.count)**2 -
                                         (parent.av_counts[attr][val] /
                                          parent.count)**2)
        
        return category_utility

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
        Given a name_assignemnts (a dict) rename the 
        components and relations and return the renamed
        instance.

        To ensure there isn't a renaming collision the mapping should be a
        complete mapping for the given instance (all components should
        have a name they map to). Will throw assertion error if this is not
        true.
        """
        # Ensure it is a complete mapping
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
            score = self._reduced_category_utility(self._rename(instance,
                                                                mapping))
            scored_mappings.append((score, mapping))

        best_mapping = sorted(scored_mappings, key=lambda x: x[0])[0][1]
        print('###########')
        print(from_name)
        print(to_name)
        print(best_mapping)
        print('###########')
        return best_mapping

    def _no_match(self, instance):
        """
        Don't do any matching, just rename things to themselves.
        """
        mapping = {}
        for attr in instance:
            if isinstance(instance[attr], Labyrinth):
                mapping[attr] = attr

        return mapping

    def _match(self, instance):
        """ 
        Define the specialized matching function to rename components
        and relations to maximize the match between instance and the
        current concept (self).
        """
        mapping = self._exhaustive_match(instance)
        temp_instance = self._rename(instance, mapping)
        return temp_instance

    def _labyrinth_categorize(self, instance):
        temp_instance = {}

        for attr in instance:
            for val in instance[attr]:
                if isinstance(val, dict):
                    temp_instance[attr] = self._labyrinth_categorize(val)
                else:
                    temp_instance[attr] = val

        return self._cobweb_categorize(temp_instance)

    def _category_utility(self):
        if len(self.children) == 0:
            return 0.0

        category_utility = 0.0

        for child in self.children:
            p_of_child = child.count / self.count
            category_utility += (p_of_child *
                                 (child._category_utility_nominal(self) +
                                  child._category_utility_numeric(self) +
                                  child._category_utility_component(self)))
        return category_utility / (1.0 * len(self.children))

    def ifit(self, instance):
        self._labyrinth(instance)

    def _replace(self, old, new):
        """
        Traverse the tree and replace all references to concept old with
        concept new.
        """
        temp_counts = {}
        for attr in self.av_counts:
            temp_counts[attr] = {}
            for val in self.av_counts[attr]:
                x = val
                if val == old:
                    x = new
                if x not in temp_counts[attr]:
                    temp_counts[attr][x] = 0
                temp_counts[attr][x] += self.av_counts[attr][val] 

        self.av_counts = temp_counts

        for c in self.children:
            c._replace(old,new)

    def _split(self, best):
        """
        Specialized version of split for labyrinth. This removes all references
        to a particular concept from the tree. It replaces these references
        with a reference to the parent concept
        """
        self.children.remove(best)
        for child in best.children:
            self.children.append(child)

        # replace references to deleted concept with parent concept
        self.__class__.root._replace(best, self)

    def _pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        tabs = "\t" * depth
        ret = str(('\t' * depth) + "|-" + "[" + self.concept_name + "]\n" +
                  tabs + "  ")

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
                  
        ret += "{" + (",\n" + tabs + "   ").join(attributes) + "}: " + str(self.count) + '\n'
        
        for c in self.children:
            ret += c._pretty_print(depth+1)

        return ret

if __name__ == "__main__":

    t = Labyrinth()

    t.train_from_json("labyrinth_test.json")
    print(t)

    test = {}
    test['sample_mean'] = "40"
    #print(t.predict(test))



