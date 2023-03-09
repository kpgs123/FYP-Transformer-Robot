import math

maze = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1,0,1],
        [1,0,1,0,0,0,0,0,0,1],
        [1,0,1,0,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]
               
r, c = len(maze), len(maze[0])

start = (1, 2)
end = (4, 3)

sq = math.sqrt(2)

s_e = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

dic = {}

for i in range(r):
	for j in range(c):
		if not maze[i][j]:
			dic[(i,j)] = [float('inf'), float('inf'), float('inf'), start]
			
dic[start] =  [0, s_e, s_e, start]

print(dic)

while True:
    dic = dict(sorted(dic.items(), key=lambda x:x[1][2]))
    q = list(dic.keys())[0]
    print(q)
    if q == end:
        break
    i, j = q
    cond = False
    if (i - 1, j - 1) in dic.keys():
        if dic[(i - 1, j - 1)][0] > dic[q][0] + sq:
            dic[(i - 1, j - 1)][0] = dic[q][0] + sq
            cond = True
    elif i - 1 >= 0 and j - 1 >= 0 and not maze[i - 1][j - 1]:
        g = dic[q][0] + sq
        h = math.sqrt((end[0] - (i - 1))**2 + (end[1] - (j - 1))**2)
        f = g + h
        dic[i - 1, j - 1] = [g, h, f, q]
        
    if (i -1, j) in dic.keys():
        if dic[(i - 1, j)][0] > dic[q][0] + 1:
            dic[(i - 1, j)][0] = dic[q][0] + 1
            cond = True
    elif i - 1 >= 0 and not maze[i - 1][j]:
        g = dic[q][0] + 1
        h = math.sqrt((end[0] - (i - 1))**2 + (end[1] - (j))**2)
        f = g + h
        dic[i - 1, j] = [g, h, f, q]

    if (i - 1, j + 1) in dic.keys():
        if dic[(i - 1, j + 1)][0] > dic[q][0] + sq:
            dic[(i - 1, j + 1)][0] = dic[q][0] + sq
            cond = True
    elif i - 1 >= 0 and j + 1 < c and not maze[i - 1][j + 1]:
        g = dic[q][0] + sq
        h = math.sqrt((end[0] - (i - 1))**2 + (end[1] - (j + 1))**2)
        f = g + h
        dic[i - 1, j + 1] = [g, h, f, q]

    if (i, j + 1) in dic.keys():
        if dic[(i, j + 1)][0] > dic[q][0] + 1:
            dic[(i, j + 1)][0] = dic[q][0] + 1
            cond = True
    elif j + 1 < c and not maze[i][j + 1]:
        g = dic[q][0] + 1
        h = math.sqrt((end[0] - (i))**2 + (end[1] - (j + 1))**2)
        f = g + h
        dic[i, j + 1] = [g, h, f, q]

    if (i + 1, j + 1) in dic.keys():
        if dic[(i + 1, j + 1)][0] > dic[q][0] + sq:
            dic[(i + 1, j + 1)][0] = dic[q][0] + sq
            cond = True
            print(dic[(i +1, j + 1)])
    elif i + 1 < r and j + 1 < c and not maze[i + 1][j + 1]:
        g = dic[q][0] + sq
        h = math.sqrt((end[0] - (i + 1))**2 + (end[1] - (j + 1))**2)
        f = g + h
        dic[i + 1, j + 1] = [g, h, f, q]

    if (i + 1 ,j) in dic.keys():
        if dic[(i + 1, j)][0] > dic[q][0] + 1:
            dic[(i + 1, j)][0] = dic[q][0] + 1
            cond = True
    elif i + 1 < r and not maze[i + 1][j]:
        g = dic[q][0] + 1
        h = math.sqrt((end[0] - (i + 1))**2 + (end[1] - (j))**2)
        f = g + h
        dic[i + 1, j] = [g, h, f, q]

    if (i + 1, j - 1) in dic.keys():
        if dic[(i + 1, j - 1)][0] > dic[q][0] + sq:
            dic[(i + 1, j - 1)][0] = dic[q][0] + sq
            cond = True
    elif i + 1 < r and j - 1 >= 0 and not maze[i + 1][j - 1]:
        g = dic[q][0] + sq
        h = math.sqrt((end[0] - (i + 1))**2 + (end[1] - (j - 1))**2)
        f = g + h
        dic[i + 1, j - 1] = [g, h, f, q]

    if (i, j - 1) in dic.keys():
        if dic[(i, j - 1)][0] > dic[q][0] + 1:
            dic[(i, j - 1)][0] = dic[q][0] + 1
            cond = True
    elif j - 1 >= 0 and not maze[i][j - 1]:
        g = dic[q][0] + 1
        h = math.sqrt((end[0] - (i))**2 + (end[1] - (j - 1))**2)
        f = g + h
        dic[i, j - 1] = [g, h, f, q]

    if not cond:
        dic[q][2] = float('inf')
    break