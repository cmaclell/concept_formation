from trestle import Trestle
import copy
import random

class SimStudent:

    def __init__(self):
        self.memory = Trestle()

    def suggest_action(self, state):
        """
        Given a state, this function finds the best concept to represent the
        state, then uses this concept to identify the most applicable action to
        take-- elaborating the state where necessary to take action.
        """
        instance = copy.deepcopy(state)
        instance['outcome'] = "CORRECT"
        concept = self.memory.trestle_categorize(instance)
        #print(concept)

        while concept:
            possible = []
            for attr in concept.av_counts:
                if isinstance(attr, tuple) and attr[0] == 'action':
                    possible += [attr] * concept.av_counts[attr][True]
            if possible:
                action = random.choice(possible)

                new_action = []
                for i,v in enumerate(action):
                    if i == 0:
                        new_action.append(action[i])
                        continue
                    
                    values = []
                    for val in concept.av_counts[v]:
                        #print(self.memory)
                        if 'value' in val.av_counts:
                            for x in val.av_counts['value']:
                                values += [x] * val.av_counts['value'][x] * concept.av_counts[v][val]

                    new_action.append(random.choice(values))

                return new_action
            else:
                concept = concept.parent

    def demonstration(self, state, action, outcome):
        """
        Incorporate a demonstration into memory. If the system suggested an
        action and received feedback, then it is incorporated here with the
        outcome. 
        """ 
        instance = copy.deepcopy(state)
        instance['outcome'] = outcome
        concept = self.memory.trestle_categorize(instance)

        new_action = []
        for i,v in enumerate(action):
            if i == 0:
                new_action.append(action[i])
                continue

            match = False
            for attr in instance:
                if 'value' in instance[attr] and instance[attr]['value'] == v:
                    print("matched in instance")
                    new_action.append(attr)
                    match = True
                    break

            #if not match:
            #    for attr in concept.av_counts:
            #        values = []
            #        for val in concept.av_counts[attr]:
            #            if isinstance(val, Trestle) and 'value' in val.av_counts:
            #                #print(val.av_counts['value'])
            #                for val2 in val.av_counts['value']:
            #                    values += [val2] * val.av_counts['value'][val2] * concept.av_counts[attr][val]
            #        if values and random.choice(values) == v:
            #            print("matching old")
            #            instance[attr] = {'value': v}
            #            new_action.append(attr)
            #            match = True
            #            break

            if not match:
                print("Generating new symbol")
                name = "component" + self.memory.gensym()
                # TODO better way to do this?
                while name in instance:
                    name = "component" + self.memory.gensym()

                instance[name] = {'value': v}
                new_action.append(name)
        
        instance['action-relation'] = new_action
        self.memory.trestle(instance)

if __name__ == "__main__":
    ss = SimStudent()

    #s0 = {'s': {'value': "3"}}
    #action0 = ['action', 'speak', 'number']

    #s1 = {'s': {'value': "5"}}
    #action1 = ['action', 'speak', 'number']

    #s2 = {'s': {'value': 'a'}}
    #action2 = ['action', 'speak', 'letter']

    #s3 = {'o1': {'value': "3"},
    #      'o2': {'value': "x"},
    #      'r1': ['before', 'o1', 'o2']}
    #action3 = ['action', 'divide', 'number']

    #s4 = {'o1': {'value': "a"},
    #      'o2': {'value': "x"},
    #      'r1': ['before', 'o1', 'o2']}
    #action4 = ['action', 'divide', 'letter']

    #s5 = {'s': {'value': "10"}}
    #action5 = ['action', 'speak', 'number']

    #s6 = {'o1': {'value': "10"},
    #      'o2': {'value': "x"},
    #      'r1': ['before', 'o1', 'o2']}
    #action6 = ['action', 'divide', 'number']
    #
    #s7 = {'s': {'value': "20"}}
    #action7 = ['action', 'speak', 'number']

    #s8 = {'o1': {'value': "20"},
    #      'o2': {'value': "x"},
    #      'r1': ['before', 'o1', 'o2']}
    #action8 = ['action', 'divide', 'number']

    #print(s0, action0)
    #print(ss.suggest_action(s0))
    #ss.demonstration(s0, action0, "CORRECT")
    #print(ss.suggest_action(s0))

    #print("--------------")

    #print(s1, action1)
    #print(ss.suggest_action(s1))
    #ss.demonstration(s1, action1, "CORRECT")
    #print(ss.suggest_action(s1))

    #print('--------------')

    #print(s2, action2)
    #print(ss.suggest_action(s2))
    #ss.demonstration(s2, action2, "CORRECT")
    #print(ss.suggest_action(s2))

    #print('--------------')

    #print(s3, action3)
    #print(ss.suggest_action(s3))
    #ss.demonstration(s3, action3, "CORRECT")
    #print(ss.suggest_action(s3))

    #print('--------------')

    #print(s4, action4)
    #print(ss.suggest_action(s4))
    #ss.demonstration(s4, action4, "CORRECT")
    #print(ss.suggest_action(s4))

    #print('--------------')

    #print(s5, action5)
    #print(ss.suggest_action(s5))
    #ss.demonstration(s5, action5, "CORRECT")
    #print(ss.suggest_action(s5))

    #print('--------------')

    #print(s6, action6)
    #print(ss.suggest_action(s6))
    #ss.demonstration(s6, action6, "CORRECT")
    #print(ss.suggest_action(s6))

    #print('--------------')

    #print(s7, action7)
    #print(ss.suggest_action(s7))
    #ss.demonstration(s7, action7, "CORRECT")
    #print(ss.suggest_action(s7))

    #print('--------------')

    #print(s8, action8)
    #print(ss.suggest_action(s8))
    #ss.demonstration(s8, action8, "CORRECT")
    #print(ss.suggest_action(s8))

    #print(ss.memory.trestle_categorize({'o1': {'value': '10'}}))
    
    #print(ss.memory)
    
    for i in range(10):
        test = {}
        test['o1'] = {'type': 'label', 'left': 5.0, 'y': 20.0}
        test['o2'] = {'value': 'Find'}
        test['o3'] = {'value': 'the'}
        test['o4'] = {'value': 'square'}
        test['o5'] = {'value': 'of'}
        test['r1'] = ['in', 'o2', 'o1']
        test['r2'] = ['in', 'o3', 'o1']
        test['r3'] = ['in', 'o4', 'o1']
        test['r4'] = ['in', 'o5', 'o1']
        test['r5'] = ['before', 'o2', 'o3']
        test['r6'] = ['before', 'o3', 'o4']
        test['r7'] = ['before', 'o4', 'o5']
        test['o6'] = {'type': 'input', 'left': 35.0, 'y': 18.0}

        #test['o7'] = {'value': '3'}
        #test['o8'] = {'value': '5'}

        #test['r8'] = ['in', 'o7', 'o6']
        #test['r9'] = ['in', 'o8', 'o6']
        #test['r10'] = ['before', 'o7', 'o8']
        action = ['action', 'output', 'box1',
                  random.randint(0,9), random.randint(0,9)]

        print(test, action)
        print(ss.suggest_action(test))
        ss.demonstration(test, action, "CORRECT")
        print(ss.suggest_action(test))
