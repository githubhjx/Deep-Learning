r = open('/home/s2/data/Pain/label.txt')
file = r.readlines()
records = []

for i in range(len(file)):
    f = file[i].split()
    ft = list(map(int, f))
    records.append(ft)

f = open('/home/s2/data/Pain/test.txt', 'w')
for i in range(len(records)):
    for j in range(len(records[i])):
        f.write(str(records[i][j]))
        f.write(' ')
    f.write('\n')


# f.close()
# print(r)


r = open('/home/s2/data/Pain/test.txt')
file = r.readlines()
records = []
# file = open(path)
# file = file.readlines()
for i in range(len(file)):
    f = file[i].split()
    ft = list(map(int, f))
    records.append(ft)
