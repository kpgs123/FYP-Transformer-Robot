import math

with open("maze123.txt", 'r') as file:
    s = file.read()
    maze = eval(s)
               
r, c = len(maze), len(maze[0])

start = (0, 0)
end = (24, 24)

sq = math.sqrt(2)

s_e = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

dic = {}
final_dic = {}

for i in range(r):
	for j in range(c):
		if not maze[i][j]:
			dic[(i,j)] = [float('inf'), float('inf'), float('inf'), start]
			
dic[start] =  [0, s_e, s_e, start]

while True:
    dic = dict(sorted(dic.items(), key=lambda x:x[1][2]))
    q = list(dic.keys())[0]
    i, j = q
    cond = False
    final_dic[q] = dic[q]
    if q == end:
        break
    if i - 1 >= 0 and j - 1 >= 0:
        if (i - 1, j - 1) in dic.keys():
            if dic[(i - 1, j - 1)][0] > dic[q][0] + sq:
                g = dic[q][0] + sq
            else:
                g = dic[(i - 1, j - 1)][0]
            h = math.sqrt((end[0] - (i - 1))**2 + (end[1] - (j - 1))**2)
            f = g + h
            dic[i - 1, j - 1] = [g, h, f, q]
    if i - 1 >= 0:
        if (i -1, j) in dic.keys():
            if dic[(i - 1, j)][0] > dic[q][0] + 1:
                g = dic[q][0] + 1
            else:
                g = dic[(i - 1, j)][0]
            g = dic[q][0] + 1
            h = math.sqrt((end[0] - (i - 1))**2 + (end[1] - (j))**2)
            f = g + h
            dic[i - 1, j] = [g, h, f, q]

    if i - 1 >= 0 and j + 1 < c:
        if (i - 1, j + 1) in dic.keys():
            if dic[(i - 1, j + 1)][0] > dic[q][0] + sq:
                g = dic[q][0] + sq
            else:
                g = dic[(i - 1, j + 1)][0]
            h = math.sqrt((end[0] - (i - 1))**2 + (end[1] - (j + 1))**2)
            f = g + h
            dic[i - 1, j + 1] = [g, h, f, q]

    if j + 1 < c:
        if (i, j + 1) in dic.keys():
            if dic[(i, j + 1)][0] > dic[q][0] + 1:
                g = dic[q][0] + 1
            else:
                g = dic[(i, j + 1)][0]
            h = math.sqrt((end[0] - (i))**2 + (end[1] - (j + 1))**2)
            f = g + h
            dic[i, j + 1] = [g, h, f, q]

    if i + 1 < r and j + 1 < c:
        if (i + 1, j + 1) in dic.keys():
            if dic[(i + 1, j + 1)][0] > dic[q][0] + sq:
                g = dic[q][0] + sq
            else:
                g = dic[(i + 1, j + 1)][0]
            h = math.sqrt((end[0] - (i + 1))**2 + (end[1] - (j + 1))**2)
            f = g + h
            dic[i + 1, j + 1] = [g, h, f, q]

    if i + 1 < r:
        if (i + 1 ,j) in dic.keys():
            if dic[(i + 1, j)][0] > dic[q][0] + 1:
                g = dic[q][0] + 1
            else:
                g = dic[(i + 1, j)][0]
            h = math.sqrt((end[0] - (i + 1))**2 + (end[1] - (j))**2)
            f = g + h
            dic[i + 1, j] = [g, h, f, q]

    if i + 1 < r and j - 1 >= 0:
        if (i + 1, j - 1) in dic.keys():
            if dic[(i + 1, j - 1)][0] > dic[q][0] + sq:
                g = dic[q][0] + sq
            else:
                g = dic[(i + 1, j - 1)][0]
            h = math.sqrt((end[0] - (i + 1))**2 + (end[1] - (j - 1))**2)
            f = g + h
            dic[i + 1, j - 1] = [g, h, f, q]

    if j - 1 >= 0:
        if (i, j - 1) in dic.keys():
            if dic[(i, j - 1)][0] > dic[q][0] + 1:
                g = dic[q][0] + 1
            else:
                g = dic[(i, j - 1)][0]
            h = math.sqrt((end[0] - (i))**2 + (end[1] - (j - 1))**2)
            f = g + h
            dic[i, j - 1] = [g, h, f, q]
    
    del dic[q]

lst = [end]
k = end
while k != start:
    k = final_dic[k][3]
    lst.append(k)

lst = lst[::-1]

with open("solved_maze.txt", 'w') as file:
    for row in range(r):
        s = ""
        for column in range(c):
            square = maze[row][column]
            if (row, column) in lst:
                s += '+'
            elif square == 1:
                s += '%'
            else:
                s += '&'
            s += ' '
        file.write(s+'\n')
print(lst)