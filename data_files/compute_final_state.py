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

if __name__ == "__main__":

    endStates = {}
    allStates = {}

    input_file = '40-students.txt'
    output_intermediate = 'instant-test-intermediate.txt'
    output_final = 'instant-test-processed.txt'

    with open(output_intermediate, 'w') as outputfile:
        writer = csv.writer(outputfile, delimiter='\t')
        file_name = input_file
        with open(file_name, newline='') as csvfile:
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

                allStates[row[key['Problem Name']]].append(json.loads(row[key['Selection']]))
                writer.writerow(row)
                previous = row

    count = 0
    previous = None
    with open(output_final, 'w') as outputfile:
        writer = csv.writer(outputfile, delimiter='\t')
        file_name = output_intermediate
        with open(file_name, newline='') as csvfile:
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
                for i,s in enumerate(allStates[row[key['Problem Name']]]):
                    if stateMatch(s, json.loads(row[key['Selection']])):
                        row[key['Step Name']] = row[key['Problem Name']] + "_s" + str(i)
                        break

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
            writer.writerow(previous)

