from trestle import Trestle
import copy

class SimStudent:

    def __init__(self):
        self.memory = Trestle()

    def suggest_action(self, state):
        """
        Given a state, this function finds the best concept to represent the
        state, then uses this concept to identify the most applicable action to
        take-- elaborating the state where necessary to take action.
        """
        concept = self.memory.trestle_categorize(state)
        print(concept)


    def demonstration(self, state, action, outcome):
        """
        Incorporate a demonstration into memory. If the system suggested an
        action and received feedback, then it is incorporated here with the
        outcome. 
        """ 
        instance = copy.deepcopy(state)
        # first explain the action, elaborating the state as necessary

        # second, incorporate into memory
        self.memory.trestle(instance)

if __name__ == "__main__":
    ss = SimStudent()

    
