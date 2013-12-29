from labyrinth import Labyrinth

class Trestle(Labyrinth):

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

    def _probability_given(self, other):
        """
        The probability that the current node would be reached from another
        provided node.
        """
        probs = [((child.count / (other.count * 1.0)) *
                 self._probability_given(child)) for child in other.children]
        if len(probs) == 0:
            return 0.0
        
        return max(probs)

    def _expected_correct_guesses(self):
        """
        Computes the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values.
        """
        # acuity the smallest allowed standard deviation; default = 1.0 
        acuity = 1.0
        correct_guesses = 0.0

        for attr in self.av_counts:
            float_values = []
            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    float_values += [val] * int(self.av_counts[attr][val])
                else:
                    if isinstance(val, Labyrinth):
                        prob = 0.0
                        for val2 in self.av_counts[attr]:
                            if isinstance(val2, Labyrinth):
                                prob += (((1.0 * self.av_counts[attr][val2]) /
                                          self.count) *
                                         val._probability_given(val2))
                    else:
                        prob = ((1.0 * self.av_counts[attr][val]) / self.count)
                    correct_guesses += (prob * prob)

            # handle the float values
            if len(float_values) == 0:
                continue
            std = self._std(float_values)
            if std < acuity:
                std = acuity
            correct_guesses += (1.0 / (2.0 * math.sqrt(math.pi) * std))

        return correct_guesses

if __name__ == "__main__":

    t = Trestle()
    t.train_from_json("labyrinth_test.json")
    t.verify_counts()
    print(t)

    test = {}
    print(t.predict(test))

