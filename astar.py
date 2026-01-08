import pygame
import math
from math import sqrt
from queue import PriorityQueue
from timeit import default_timer as timer

heuristic_func = None

# pygame window for visualisation
WIDTH = 800
WINDOW = pygame.display.set_mode((WIDTH, WIDTH)) # width == height
pygame.display.set_caption("A* Pathfinding Algorithm")

# pygame colour codes
TAN = (230,220,170)
MAROON = ((115,0,0))
COFFEE_BROWN = (200, 190, 140)
BLACK = (0, 0, 0)
MOON_GLOW = (235,245,255)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
PURPLE = (128, 0, 128)

class Node:
	'''Class for individual cube on grid'''
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = COFFEE_BROWN # start colour of grid
		self.neighbours = []
		self.width = width
		self.total_rows = total_rows

	# methods to get state of node
	def get_pos(self):
		return self.row, self.col

	def is_closed(self):
		return self.color == TAN

	def is_open(self):
		return self.color == MAROON

	# obstacle in path
	def is_obstacle(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == PURPLE

	def reset(self):
		self.color = COFFEE_BROWN

	# methods to update state of node
	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = TAN

	def make_open(self):
		self.color = MAROON

	def make_obstacle(self):
		self.color = BLACK

	def make_end(self):
		self.color = PURPLE

	def make_path(self):
		self.color = MOON_GLOW

	def draw(self, window):
		# draws node
		pygame.draw.rect(window, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbours(self, grid):
		# update the neighbouring nodes 
		self.neighbours = []
		if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle(): # up
			self.neighbours.append(grid[self.row - 1][self.col])

		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_obstacle(): # down
			self.neighbours.append(grid[self.row + 1][self.col])

		if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle(): # left
			self.neighbours.append(grid[self.row][self.col - 1])

		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_obstacle(): # right
			self.neighbours.append(grid[self.row][self.col + 1])


	# less than operator to check that cube is less than next so can move
	def __lt__(self, other):
		return False

def manhattan(p1, p2):
	# manhattan heuristic
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

def euclidean(p1, p2):
	# euclidean heuristic
	x1, y1 = p1
	x2, y2 = p2
	return sqrt((x1-x2)**2+(y1-y2)**2)


def reconstruct_path(came_from, current, draw):
	path_length = 0 # amount of nodes in shortest path
	while current in came_from:
		path_length += 1
		current = came_from[current]
		current.make_path()
		draw()
	print("Shortest path of " + str(path_length) + " nodes has been found")


def algorithm(draw, grid, start, end, heuristic):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start)) # add start node with original abs score (0) to open set
	came_from = {} # dict of previous nodes (path)
	g_score = {node: float("inf") for row in grid for node in row} # dict of g score for each node
	g_score[start] = 0
	f_score = {node: float("inf") for row in grid for node in row}
	f_score[start] = heuristic(start.get_pos(), end.get_pos()) # f score will be the heuristic to estimate distance at beginning

	open_set_hash = {start} # check if node is in priorityqueue

	# algorithm runs until all nodes closed
	print("Finding shortest path...")
	start_time = timer()
	while not open_set.empty():
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = open_set.get()[2] # index at 2 to get the node from priorityqueue
		open_set_hash.remove(current) # remove lowest value f score node

		if current == end: # found path
			end_time = timer()
			reconstruct_path(came_from, end, draw)
			start.make_start()
			end.make_end()
			print("Algorithm executed in " + str(end_time - start_time) + " secs")
			return True

		for neighbour in current.neighbours:
			temp_g_score = g_score[current] + 1

			if temp_g_score < g_score[neighbour]: # if found a better way to reach neighbour
				came_from[neighbour] = current
				g_score[neighbour] = temp_g_score
				f_score[neighbour] = temp_g_score + heuristic(neighbour.get_pos(), end.get_pos())
				if neighbour not in open_set_hash: # check if neighbour in set
					count += 1
					open_set.put((f_score[neighbour], count, neighbour))
					open_set_hash.add(neighbour)
					neighbour.make_open()

		draw()

		if current != start:
			current.make_closed()

	print ("No path has been found")
	return False # did not find path


def make_grid(rows, width):
	grid = []
	gap = width // rows # width of cubes
	# 2d list of nodes [[], [], []]
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			node = Node(i, j, gap, rows)
			grid[i].append(node)

	return grid


def draw_grid(window, rows, width):
	# draw grid lines
	gap = width // rows
	for i in range(rows):
		pygame.draw.line(window, GREY, (0, i * gap), (width, i * gap)) # draw horizontal line fo each row
		for j in range(rows):
			pygame.draw.line(window, GREY, (j * gap, 0), (j * gap, width)) # draw vertical line fo each column


def draw(window, grid, rows, width):
	# draw grid with nodes
	window.fill(COFFEE_BROWN)

	for row in grid:
		for node in row:
			node.draw(window)

	draw_grid(window, rows, width)
	pygame.display.update()


def get_clicked_pos(pos, rows, width):
	# translate mouse position to cube on grid
	gap = width // rows
	y, x = pos

	row = y // gap
	col = x // gap

	return row, col

print("Choose heuristic:")
print("1 = manhattan")
print("2 = euclidean")

def choose_heuristic():
	choice = input("Enter 1 or 2: ")

	if choice == "1":
		print ("Using manhattan heuristic")
		return manhattan
	else:
		print("Using euclidean heuristic")
		return euclidean

def main(window, width):
    global heuristic_func

    ROWS = 50
    grid = make_grid(ROWS, width)

    # instructions on terminal for user
    print(
        """
        A* Algorithm — Instructions
        ---------------------------

        1. Select a start node and an end node.
        2. Add obstacles to the grid.
        3. Right-click cubes to reset them.
        4. Press SPACE to begin.
        """
    )

    start = None
    end = None
    need_new_heuristic = False  # <-- flag to ask again after clearing

    running = True
    while running:
        for event in pygame.event.get():
            # stop running when window closed
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:  # left mouse click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.make_start()

                elif not end and node != start:
                    end = node
                    end.make_end()

                # after start and end nodes make obstacles with mouse
                elif node != end and node != start:
                    node.make_obstacle()

            elif pygame.mouse.get_pressed()[2]:  # right mouse click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()  # reset node with right click
                if node == start:
                    start = None
                elif node == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                # press spacebar to start algorithm
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)

                    algorithm(lambda: draw(window, grid, ROWS, width),
                              grid, start, end, heuristic_func)
                    print("Press C to clear the grid")

                # press 'c' to clear grid
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)  # <-- grid is cleared now
                    need_new_heuristic = True      # <-- ask for heuristic later

        # draw AFTER handling events so updates (like clearing) are visible immediately
        draw(window, grid, ROWS, width)

        # if the user pressed C, now ask them which heuristic to use
        if need_new_heuristic:
            heuristic_func = choose_heuristic()
            need_new_heuristic = False

    pygame.quit()


heuristic_func = choose_heuristic()
main(WINDOW, WIDTH)