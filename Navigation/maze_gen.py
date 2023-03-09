from random import randint

def notX(x):
    return not x

# Class to define structure of a node
class Node:
    def __init__(self, value = None, 
               next_element = None):
        self.val = value
        self.next = next_element
  
# Class to implement a stack
class stack:
    # Constructor
    def __init__(self):
        self.head = None
        self.length = 0
  
    # Put an item on the top of the stack
    def insert(self, data):
        self.head = Node(data, self.head)
        self.length += 1
  
    # Return the top position of the stack
    def pop(self):
        if self.length == 0:
            return None
        else:
            returned = self.head.val
            self.head = self.head.next
            self.length -= 1
            return returned
  
    # Return False if the stack is empty 
    # and true otherwise
    def not_empty(self):
        return bool(self.length)
  
    # Return the top position of the stack
    def top(self):
        return self.head.val
  
# Function to generate the random maze
def random_maze_generator(r, c, P0, Pf):
    ROWS, COLS = r, c
      
    # Array with only walls (where paths will 
    # be created)
    maze = list(list(0 for _ in range(COLS)) 
                       for _ in range(ROWS))
      
    # Auxiliary matrices to avoid cycles
    seen = list(list(False for _ in range(COLS)) 
                           for _ in range(ROWS))
    previous = list(list((-1, -1) 
     for _ in range(COLS)) for _ in range(ROWS))
  
    S = stack()
      
    # Insert initial position
    S.insert(P0) 
  
    # Keep walking on the graph using dfs
    # until we have no more paths to traverse 
    # (create)
    while S.not_empty():
  
        # Remove the position of the Stack
        # and mark it as seen
        x, y = S.pop()
        seen[x][y] = True
  
        # Check if it will create a cycle
        # if the adjacent position is valid 
        # (is in the maze) and the position 
        # is not already marked as a path 
        # (was traversed during the dfs) and 
        # this position is not the one before it
        # in the dfs path it means that 
        # the current position must not be marked.
          
        # This is to avoid cycles with adj positions
        if (x + 1 < ROWS) and maze[x + 1][y] == 1 \
        and previous[x][y] != (x + 1,  y):
            continue
        if (0 < x) and maze[x-1][y] == 1 \
        and previous[x][y] != (x-1,  y):
            continue
        if (y + 1 < COLS) and maze[x][y + 1] == 1 \
        and previous[x][y] != (x, y + 1):
            continue
        if (y > 0) and maze[x][y-1] == 1 \
        and previous[x][y] != (x, y-1):
            continue
  
        # Mark as walkable position
        maze[x][y] = 1
  
        # Array to shuffle neighbours before 
        # insertion
        to_stack = []
  
        # Before inserting any position,
        # check if it is in the boundaries of 
        # the maze
        # and if it were seen (to avoid cycles)
  
        # If adj position is valid and was not seen yet
        if (x + 1 < ROWS) and seen[x + 1][y] == False:
              
            # Mark the adj position as seen
            seen[x + 1][y] = True
              
            # Memorize the position to insert the 
            # position in the stack
            to_stack.append((x + 1,  y))
              
            # Memorize the current position as its 
            # previous position on the path
            previous[x + 1][y] = (x, y)
          
        if (0 < x) and seen[x-1][y] == False:
              
            # Mark the adj position as seen
            seen[x-1][y] = True
              
            # Memorize the position to insert the 
            # position in the stack
            to_stack.append((x-1,  y))
              
            # Memorize the current position as its 
            # previous position on the path
            previous[x-1][y] = (x, y)
          
        if (y + 1 < COLS) and seen[x][y + 1] == False:
              
            # Mark the adj position as seen
            seen[x][y + 1] = True
              
            # Memorize the position to insert the 
            # position in the stack
            to_stack.append((x, y + 1))
              
            # Memorize the current position as its
            # previous position on the path
            previous[x][y + 1] = (x, y)
          
        if (y > 0) and seen[x][y-1] == False:
              
            # Mark the adj position as seen
            seen[x][y-1] = True
              
            # Memorize the position to insert the 
            # position in the stack
            to_stack.append((x, y-1))
              
            # Memorize the current position as its 
            # previous position on the path
            previous[x][y-1] = (x, y)
          
        # Indicates if Pf is a neighbour position
        pf_flag = False
        while len(to_stack):
              
            # Remove random position
            neighbour = to_stack.pop(randint(0, len(to_stack)-1))
              
            # Is the final position, 
            # remember that by marking the flag
            if neighbour == Pf:
                pf_flag = True
              
            # Put on the top of the stack
            else:
                S.insert(neighbour)
          
        # This way, Pf will be on the top 
        if pf_flag:
            S.insert(Pf)
                  
    # Mark the initial position
    x0, y0 = P0
    xf, yf = Pf
    maze[x0][y0] = 2
    maze[xf][yf] = 3
      
    # Return maze formed by the traversed path
    return maze

def andOfTwoLists(list1, list2):
    new_lst = []
    r = len(list1)
    c = len(list1[0])
    for i in range(r):
        row_lst = []
        for j in range(c):
            row_lst.append(list1[i][j] or list2[i][j])
        new_lst.append(row_lst)
    return new_lst
  
# Driver code
if __name__ == "__main__":
    N = 25
    M = 25
    P0 = (0, 0)
    P1 = (24, 24)
    maze = random_maze_generator(N, M, P0, P1)
    #maze2 = random_maze_generator(N, M, P0, P1)
    #maze = andOfTwoLists(maze1, maze2)
    maze[P0[0]][P0[1]] = 1
    maze[P1[0]][P1[1]] = 1
    maze2 = []
    for line in maze:
        maze2.append(list(map(notX, line)))
    with open("maze123.txt", 'w') as file:
        file.writelines(str(maze2))