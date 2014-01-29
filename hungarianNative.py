import munkres

def hungarian(costMatrix) :
    hun = munkres.Munkres()
    map = hun.compute(costMatrix)
    ret = {}
    for tup in map:
        print tup[0]
        ret[tup[0]] = tup[1]
    print ret
    return ret

if __name__ == '__main__' :
    matrix = [[400, 150, 400],
              [400, 450, 600],
              [300, 225, 300]]

    print hungarian(matrix)
