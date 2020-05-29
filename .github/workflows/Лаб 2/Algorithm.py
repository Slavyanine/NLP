from pyjarowinkler import distance

def Heming(str1, str2):
    d = len(str1)-len(str2)
    return (abs(d)+sum(i != j for i, j in zip(s1, s2)))/max(len(str1), len(str2))


def Jaro(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    h = int(max(len(str1), len(str2)) / 2) - 1
    m, t = 0, 0
    bf = dict()
    lst1 = [0 for _ in range(len(str1))]
    lst2 = [0 for _ in range(len(str2))]
    for i in range(len(str2)):
        for k in range(i - h, i + h + 1):
            if k < 0:
                continue
            if k > len(str1) - 1:
                break
            if str1[k] == str2[i]:
                lst1[k] = str1[k]
                lst2[i] = str2[i]
                if bf.get(i) == str1[k]:
                    continue
                else:
                    bf[i] = str1[k]
                    m += 1
                    # if i!=k:
                    #   t+=1

    lst1 = [i for i in lst1 if i != 0]
    lst2 = [i for i in lst2 if i != 0]
  #  print(lst1)
   # print(lst2)
    t = int(sum(i != j for i, j in zip(lst1, lst2)) / 2)
    if m == 0:
        return 0
    else:
        return (1 / 3) * (m / len(str1) + m / len(str2) + (m - t) / m)

def diff(a,b):
    if a==b:
        return 0
    else:
        return 1

def Levinstein(str1,str2):
    D = [[0]*(len(str2)+1)for i in range(len(str1)+1)]
    for i in range(len(str1)+1):
        D[i][0] = i
    for j in range(len(str2)+1):
        D[0][j] = j
    for i in range(1,len(str1)+1):
        for j in range(1, len(str2)+1):
            c = diff(str1[i-1], str2[j-1])
            D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+c)
    return D[len(str1)][len(str2)]

s1 = "GEESE"
s2 = "CHEESE"
# print(Heming(s1,s2))
#
# print(Jaro(s1,s2))
# print(Jaro('MARTHA','MARHTA'))
# print(Jaro('DWAYNE','DUANE'))
# print(Jaro('DIXON','DICKSONX'))
# print(Jaro('АБРАКАДАБРА','АБРАКАДАБРА'))
# print(Jaro('кукушка','куксится'))
#
# print(distance.get_jaro_distance('АБАБ','БАБА', winkler=False, scaling=0.1))
#
# print(Levinstein(s1,s2))