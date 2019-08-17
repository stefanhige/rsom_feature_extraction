

a = ['a','b',['c','d',['e','f',['g']]]]


a = ['a','b',['c','d',['e','f']]]

result = [k for i in a for j in i for k in j]
print(result)


def rec(l):
    out = []
    for i in l:
        out = out + rec(i) if type(i) == list else out + [i]
    return out


print(rec(a))
# print([rec(k) for k in a])
