import pygame
from math import sqrt
from queue import PriorityQueue
from timeit import default_timer as timer

pygame.init()

# =========================================================
#                SCALE-BASED SIZE SETTINGS
# =========================================================
SCALE = 0.8  # <-- change this (e.g. 1.0, 0.8, 0.6)

BASE_GRID_WIDTH = 800

BASE_FONT_SIZE = 18
BASE_FONT_BIG_SIZE = 24

BASE_BTN_H = 36
BASE_BTN_W1 = 150
BASE_BTN_W2 = 150
BASE_BTN_W3 = 120
BASE_BTN_W4 = 120

BASE_MARGIN_X = 12
BASE_MARGIN_Y = 10
BASE_BTN_GAP = 8
BASE_BTN_RADIUS = 8
BASE_BTN_BORDER = 2

BASE_TEXT_GAP = 8
STATUS_LINE_COUNT = 4


def S(x: int) -> int:
    """Scale helper -> int pixel."""
    return max(1, int(x * SCALE))


GRID_WIDTH = S(BASE_GRID_WIDTH)

# --- Colours ---
TAN = (230, 220, 170)
MAROON = (115, 0, 0)
COFFEE_BROWN = (200, 190, 140)
BLACK = (0, 0, 0)
MOON_GLOW = (235, 245, 255)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
PURPLE = (128, 0, 128)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)

# --- Fonts (scaled) ---
FONT_PATH = 'font/NotoSansMono-VariableFont_wdth,wght.ttf'

def load_font(path, size):
    try:
        return pygame.font.Font(path, size)
    except FileNotFoundError:
        print(f"Font not found: {path}, falling back to system font")
        return pygame.font.SysFont("consolas", size)

FONT = pygame.font.Font(FONT_PATH, max(10, S(BASE_FONT_SIZE)))
FONT_BIG = pygame.font.Font(FONT_PATH, max(12, S(BASE_FONT_BIG_SIZE)))

FONT_BIG.set_bold(True)

# --- Compute UI height dynamically so text never overlaps buttons ---
TITLE_H = FONT_BIG.get_height()
LINE_H = FONT.get_height()

BTN_H = S(BASE_BTN_H)
UI_HEIGHT = (
    S(BASE_MARGIN_Y) + TITLE_H +
    S(BASE_MARGIN_Y) + (STATUS_LINE_COUNT * LINE_H) +
    S(BASE_TEXT_GAP) +
    BTN_H +
    S(BASE_MARGIN_Y)
)

WIDTH = GRID_WIDTH
HEIGHT = GRID_WIDTH + UI_HEIGHT

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding Algorithm")

heuristic_func = None


# ---------- Grid axis helpers (prevents last row/col stretching) ----------
def build_axes(rows: int, grid_width: int):
    """
    Returns pixel boundaries for each cell.
    axis[i] is the starting pixel of cell i, axis[i+1] is the end.
    Length: rows+1, axis[-1] == grid_width.
    """
    return [round(i * grid_width / rows) for i in range(rows + 1)]


def find_index(axis, value):
    """
    Find cell index such that axis[i] <= value < axis[i+1]
    axis is sorted, len = rows+1
    """
    # Linear scan is fine for <= 100 rows; avoids importing bisect.
    # You can replace with bisect for speed if needed.
    for i in range(len(axis) - 1):
        if axis[i] <= value < axis[i + 1]:
            return i
    return None


class Node:
    """Class for individual cube on grid"""
    def __init__(self, row, col, x_axis, y_axis, total_rows):
        self.row = row
        self.col = col

        # Exact pixel bounds (prevents stretching artifacts)
        self.x = x_axis[col]
        self.y = y_axis[row]
        self.w = x_axis[col + 1] - x_axis[col]
        self.h = y_axis[row + 1] - y_axis[row]

        self.color = COFFEE_BROWN
        self.neighbours = []
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_obstacle(self):
        return self.color == BLACK

    def reset(self):
        self.color = COFFEE_BROWN

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
        pygame.draw.rect(window, self.color, (self.x, self.y, self.w, self.h))

    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():  # up
            self.neighbours.append(grid[self.row - 1][self.col])
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_obstacle():  # down
            self.neighbours.append(grid[self.row + 1][self.col])
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():  # left
            self.neighbours.append(grid[self.row][self.col - 1])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_obstacle():  # right
            self.neighbours.append(grid[self.row][self.col + 1])

    def __lt__(self, other):
        return False


def manhattan(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def reconstruct_path(came_from, current, draw):
    path_length = 0
    while current in came_from:
        path_length += 1
        current = came_from[current]
        current.make_path()
        draw()
    return path_length


def algorithm(draw, grid, start, end, heuristic):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    start_time = timer()
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False, None

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            end_time = timer()
            path_len = reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            return True, {"time": end_time - start_time, "path_len": path_len}

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + heuristic(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False, None


def make_grid(rows, grid_width):
    x_axis = build_axes(rows, grid_width)
    y_axis = build_axes(rows, grid_width)

    grid = []
    for r in range(rows):
        grid.append([])
        for c in range(rows):
            node = Node(r, c, x_axis, y_axis, rows)
            grid[r].append(node)

    return grid, x_axis, y_axis


def draw_grid(window, rows, grid_width, x_axis, y_axis):
    # Vertical lines
    for x in x_axis:
        pygame.draw.line(window, GREY, (x, 0), (x, grid_width))
    # Horizontal lines
    for y in y_axis:
        pygame.draw.line(window, GREY, (0, y), (grid_width, y))


def get_clicked_pos(pos, rows, grid_width, x_axis, y_axis):
    x, y = pos
    if y >= grid_width:
        return None

    col = find_index(x_axis, x)
    row = find_index(y_axis, y)
    if row is None or col is None:
        return None
    return row, col


# ---------- Simple UI widgets ----------
class Button:
    def __init__(self, rect, text, on_click):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.on_click = on_click

    def draw(self, surf, enabled=True):
        bg = (70, 70, 70) if enabled else (110, 110, 110)
        pygame.draw.rect(surf, bg, self.rect, border_radius=S(BASE_BTN_RADIUS))
        pygame.draw.rect(surf, (160, 160, 160), self.rect, S(BASE_BTN_BORDER), border_radius=S(BASE_BTN_RADIUS))
        label = FONT.render(self.text, True, WHITE)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def handle_event(self, event, enabled=True):
        if not enabled:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()


def draw_ui_bar(window, status_lines, buttons):
    ui_top = GRID_WIDTH
    ui_rect = pygame.Rect(0, ui_top, WIDTH, UI_HEIGHT)
    pygame.draw.rect(window, DARK, ui_rect)

    mx = S(BASE_MARGIN_X)
    my = S(BASE_MARGIN_Y)

    # Title
    title = FONT_BIG.render("A* Pathfinding Visualiser", True, WHITE)
    window.blit(title, (mx, ui_top + my))

    # Text block
    text_top = ui_top + my + TITLE_H + my
    y = text_top
    for line in status_lines[:STATUS_LINE_COUNT]:
        txt = FONT.render(line, True, (220, 220, 220))
        window.blit(txt, (mx, y))
        y += LINE_H

    # Buttons anchored to bottom
    btn_row_y = ui_top + UI_HEIGHT - my - BTN_H
    for b, enabled in buttons:
        b.rect.y = btn_row_y
        b.draw(window, enabled=enabled)


def draw_all(window, grid, rows, grid_width, status_lines, buttons, x_axis, y_axis):
    window.fill(COFFEE_BROWN, rect=pygame.Rect(0, 0, grid_width, grid_width))
    for row in grid:
        for node in row:
            node.draw(window)
    draw_grid(window, rows, grid_width, x_axis, y_axis)

    draw_ui_bar(window, status_lines, buttons)
    pygame.display.update()


def main():
    global heuristic_func

    ROWS = 45
    grid, x_axis, y_axis = make_grid(ROWS, GRID_WIDTH)

    start = None
    end = None

    state = "CHOOSE_HEURISTIC"
    last_result = None
    status_message = "Choose a heuristic to begin."

    def set_manhattan():
        nonlocal state, status_message
        globals()["heuristic_func"] = manhattan
        state = "EDITING"
        status_message = "Manhattan selected. Click to place Start/End, drag to add obstacles."

    def set_euclidean():
        nonlocal state, status_message
        globals()["heuristic_func"] = euclidean
        state = "EDITING"
        status_message = "Euclidean selected. Click to place Start/End, drag to add obstacles."

    def clear_grid():
        nonlocal grid, x_axis, y_axis, start, end, state, last_result, status_message
        start = None
        end = None
        last_result = None
        grid, x_axis, y_axis = make_grid(ROWS, GRID_WIDTH)
        state = "CHOOSE_HEURISTIC"
        status_message = "Grid cleared. Choose a heuristic."

    def run_astar():
        nonlocal state, last_result, status_message
        if not (start and end and globals()["heuristic_func"]):
            return

        for row in grid:
            for node in row:
                node.update_neighbours(grid)

        state = "RUNNING"
        status_message = "Running A*..."
        found, info = algorithm(
            lambda: draw_all(WINDOW, grid, ROWS, GRID_WIDTH, status_lines(), buttons_with_enabled(), x_axis, y_axis),
            grid, start, end, globals()["heuristic_func"]
        )
        if found:
            last_result = info
            status_message = f"Done. Path found: {info['path_len']} nodes, {info['time']:.4f}s"
            state = "DONE"
        else:
            status_message = "Done. No path found."
            state = "DONE"

    mx = S(BASE_MARGIN_X)
    gap = S(BASE_BTN_GAP)

    w1, w2, w3, w4 = S(BASE_BTN_W1), S(BASE_BTN_W2), S(BASE_BTN_W3), S(BASE_BTN_W4)

    x = mx
    b1 = Button((x, 0, w1, BTN_H), "Manhattan", set_manhattan)
    x += w1 + gap
    b2 = Button((x, 0, w2, BTN_H), "Euclidean", set_euclidean)
    x += w2 + gap
    b3 = Button((x, 0, w3, BTN_H), "Run", run_astar)
    x += w3 + gap
    b4 = Button((x, 0, w4, BTN_H), "Clear", clear_grid)

    def status_lines():
        heuristic_name = (
            "Manhattan" if globals()["heuristic_func"] == manhattan else
            "Euclidean" if globals()["heuristic_func"] == euclidean else
            "None"
        )
        lines = [
            f"Mode: {state}    Heuristic: {heuristic_name}",
            status_message,
            "Left-click: place Start then End. Drag: obstacles. Right-click: erase.",
        ]
        if last_result:
            lines.append(f"Last run: path_len={last_result['path_len']} time={last_result['time']:.4f}s")
        return lines

    def buttons_with_enabled():
        choose = (state == "CHOOSE_HEURISTIC")
        can_run = (state in ("EDITING", "DONE")) and start and end and globals()["heuristic_func"]
        return [
            (b1, choose),
            (b2, choose),
            (b3, bool(can_run)),
            (b4, True),
        ]

    running = True
    mouse_down_left = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            for btn, enabled in buttons_with_enabled():
                btn.handle_event(event, enabled=enabled)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down_left = True
                elif event.button == 3:
                    pos = get_clicked_pos(event.pos, ROWS, GRID_WIDTH, x_axis, y_axis)
                    if pos is not None and state in ("EDITING", "DONE"):
                        r, c = pos
                        node = grid[r][c]
                        node.reset()
                        if node == start:
                            start = None
                        elif node == end:
                            end = None

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down_left = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    run_astar()
                if event.key == pygame.K_c:
                    clear_grid()

        if state in ("EDITING", "DONE") and mouse_down_left:
            mpos = pygame.mouse.get_pos()
            pos = get_clicked_pos(mpos, ROWS, GRID_WIDTH, x_axis, y_axis)
            if pos is not None:
                r, c = pos
                node = grid[r][c]

                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != start and node != end:
                    node.make_obstacle()

        draw_all(WINDOW, grid, ROWS, GRID_WIDTH, status_lines(), buttons_with_enabled(), x_axis, y_axis)

    pygame.quit()


if __name__ == "__main__":
    main()
