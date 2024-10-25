
def get_list_from_components(components):
    #returns list of nodes of each components 
    #in a list
    l = []
    for el in components:
        l += list(el)
    return l

def map_list(liste,mapping):
    # The signaature should speak for itself :
    # (a list * (a->b) dict ) -> b  list
    return [mapping[el] for el in liste]


def parse_solution(filename):
    f = open(filename)
    s = f.readline()
    while s[:9] != "# Columns" and len(s) != 0:
        s = f.readline()
    if len(s) == 0:
        raise Exception('Infeasible')
    d = {}
    N = int(s[9:])
    for i in range(N):
        s = f.readline()
        l = s.split("(")
        try :
            l = l[:1] + l[1].split(")")
        except :
            print(l, i)
            raise Exception("Did not parse correctly")
        if not l[0] in d.keys():
            d[l[0]] = {int(l[1]) : float(l[2])}
        else:
            d[l[0]][int(l[1])] = float(l[2])
    return d
