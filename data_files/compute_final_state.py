import csv
import json

def matches(o1, o2):
    if o1 == "Inventory" or o2 == "Inventory":
        return False
    if o1['type'] != o2['type']:
        return False
    if abs(o1['x'] - o2['x']) >= 0.5:
        return False
    if abs(o1['y'] - o2['y']) >= 0.5:
        return False
    return True

def update(state, f, t):
    for o in state:
        if matches(state[o], f):
            state[o] = t
            return state
    return state

def remove_negative(s1):
    s1 = s1.copy()
    remove = []
    for name in s1:
        for v in s1[name]:
            if isinstance(s1[name][v], float) and s1[name][v] < 0:
                remove.append(name)
    for name in remove:
        del s1[name]

    return s1


def substructure(s1, s2, ufo=False):
    
    if not ufo:
        s1 = s1.copy()
        if 'UFO' in s1:
            del s1['UFO']

    # remove the negative blocks from s1 and s2
    s1 = remove_negative(s1)
    s2 = remove_negative(s2)

    for o1 in s1:
        matchSet = [o2 for o2 in s2 if matches(s1[o1], s2[o2])]
        if len(matchSet) == 0:
            return False
    return True

def stateMatch(s1, s2):
    return substructure(s1, s2, True) and substructure(s2, s1, True)

def actionMatch(a1, a2):
    if a1['from'] == "Inventory" and a2['from'] != "Inventory":
        return False
    if a2['from'] == "Inventory" and a1['from'] != "Inventory":
        return False

    if a1['action'] != a2['action']:
        return False

    if isinstance(a1['from'], dict) and isinstance(a2['from'], dict):
        if a1['from']['type'] != a2['from']['type']:
            return False
        if abs(a1['from']['x'] - a2['from']['x']) >= 0.5:
            return False
        if abs(a1['from']['y'] - a2['from']['y']) >= 0.5:
            return False
        if abs(a1['from']['rotation'] - a2['from']['rotation']) >= 10:
            return False

    if isinstance(a1['to'], dict) and isinstance(a2['to'], dict):
        if a1['to']['type'] != a2['to']['type']:
            return False
        if abs(a1['to']['x'] - a2['to']['x']) >= 0.5:
            return False
        if abs(a1['to']['y'] - a2['to']['y']) >= 0.5:
            return False
        if abs(a1['to']['rotation'] - a2['to']['rotation']) >= 10:
            return False

    return True

if __name__ == "__main__":

    endStates = {}
    allStates = {}

    input_file = '40-students.txt'
    output_intermediate = 'instant-test-intermediate.txt'
    output_final = 'instant-test-processed.txt'

    with open(output_intermediate, 'w') as outputfile:
        writer = csv.writer(outputfile, delimiter='\t')
        with open(input_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            previous = None
            key = {}

            for row in reader:
                if not key:
                    for i,v in enumerate(row):
                        key[v] = i
                    writer.writerow(row)
                    continue

                if row[key['Problem Name']] not in endStates:
                    endStates[row[key['Problem Name']]] = []
                if row[key['Problem Name']] not in allStates:
                    allStates[row[key['Problem Name']]] = []

                if row[key['Action']] == "End_State":
                    state = json.loads(previous[key['Selection']])
                    action = json.loads(previous[key['Input']])
                    #print(state)
                        #print(update(state, action['from'], action['to']))
                    newState = update(state, action['from'], action['to'])
                    if row[key['Outcome']] == "CORRECT":
                        endStates[row[key['Problem Name']]].append((newState,
                                                                    row[key['Outcome']]))
                    row[key['Selection']] = json.dumps(newState)

                if row[key['Input']]:
                    allStates[row[key['Problem Name']]].append((json.loads(row[key['Selection']]), json.loads(row[key['Input']])))
                writer.writerow(row)
                previous = row

    count = 0
    previous = None
    with open(output_final, 'w') as outputfile:
        writer = csv.writer(outputfile, delimiter='\t')
        with open(output_intermediate, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            previous = None
            key = {}
            
            for row in reader:

                count += 1
                print(count)

                if not key:
                    for i,v in enumerate(row):
                        key[v] = i
                    writer.writerow(row)
                    continue

                row[key['Step Name']] = "UNKNOWN"
                if row[key['Input']]:
                    for i,s in enumerate(allStates[row[key['Problem Name']]]):
                        state, action = s
                        #print(action)
                        if (stateMatch(state,
                                       json.loads(row[key['Selection']]))):
                            if (actionMatch(action, json.loads(row[key['Input']]))):
                                row[key['Step Name']] = row[key['Problem Name']] + "_s" + str(i)
                                break
                            else:
                                pass
                                #print("----begin----")
                                #print(action)
                                #print(json.loads(row[key['Input']]))
                                #print("----end----")

                if previous:
                    if previous[key['Action']] != "End_State":
                        state = json.loads(row[key['Selection']])
                        previous[key['Outcome']] = "INCORRECT"
                        for s,o in endStates[row[key['Problem Name']]]:
                            if substructure(state, s):
                                previous[key['Outcome']] = o
                                print(o)
                                break

                        writer.writerow(previous)

                previous = row

            #the last guy...
            if previous[key['Action']] != "End_State":
                writer.writerow(previous)

