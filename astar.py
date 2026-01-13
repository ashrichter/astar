import pygame
from math import sqrt
from queue import PriorityQueue
from timeit import default_timer as timer

pygame.init()  # Initialise all imported pygame modules

# =========================================================
#                SCALE-BASED SIZE SETTINGS
# =========================================================
# SCALE to resize the whole UI + grid consistently.
SCALE = 0.8  # <-- change this (e.g. 1.0, 0.8, 0.6)

# Base (unscaled) layout values
BASE_GRID_WIDTH = 800

BASE_FONT_SIZE = 18
BASE_FONT_BIG_SIZE = 24

BASE_BTN_H = 28
BASE_BTN_W1 = 150
BASE_BTN_W2 = 150
BASE_BTN_W3 = 120
BASE_BTN_W4 = 120

BASE_MARGIN_X = 12
BASE_MARGIN_Y = 12
BASE_BTN_GAP = 10
BASE_BTN_RADIUS = 5
BASE_BTN_BORDER = 2

# =========================================================
# CSS-like horizontal padding for tabs
# =========================================================
# These values reduce the empty space either side of button text so
# all tabs are more likely to fit on one row without widening the window.
BASE_BTN_PAD_X = 10   # NEW: left/right padding inside each button (unscaled)
BASE_BTN_MIN_W = 44   # NEW: minimum width so short labels still look like tabs

BASE_TEXT_GAP = 8
STATUS_LINE_COUNT = 4  # how many lines of status text to show in the UI bar


def S(x: int) -> int:
    """Scale helper: converts a base size into a scaled pixel size."""
    return max(1, int(x * SCALE))


# Grid width in pixels after scaling
GRID_WIDTH = S(BASE_GRID_WIDTH)

# --- Colours (some legacy / unused kept for convenience) ---
TAN = (230, 220, 170)
MAROON = (115, 0, 0)
BLACK = (0, 0, 0)
MOON_GLOW = (235, 245, 255)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
PURPLE = (128, 0, 128)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)

# =========================================================
#                     NORD COLOUR THEME
# =========================================================
# These colours define the grid background, UI bar, and node states.

# Grid & UI colours
GRID_BG_COLOR   = (76, 86, 106)      # #4C566A (dark slate background)
GRID_LINE_COLOR = (146, 154, 170)    # #929AAA (grid line colour)

UI_BG_COLOR     = (76, 86, 106)      # #4C566A (UI bar background)
UI_TEXT_COLOR   = (229, 233, 240)    # #E5E9F0 (status text)
UI_TITLE_COLOR  = (236, 239, 244)    # #ECEFF4 (title text)

# Node state colours
START_COLOR     = (163, 190, 140)    # start node
END_COLOR       = (191, 97, 106)     # end node  (#BF616A)
OBSTACLE_COLOR  = (146, 154, 170)    # obstacles (#929AAA)

OPEN_COLOR      = (129, 161, 193)    # nodes in open set
CLOSED_COLOR    = (46, 52, 64)       # nodes already explored
PATH_COLOR      = (235, 203, 139)    # final path

# =========================================================
#           TAB-LIKE (CSS-STYLE) BUTTON THEME
# =========================================================
# Button styling mimics a CSS "tab" feel with opacity changes and smooth transitions.

TAB_BG_NORMAL = GRID_LINE_COLOR         # normal tab background
TAB_BG_ACTIVE = CLOSED_COLOR            # active tab background
TAB_TEXT_NORMAL = UI_TITLE_COLOR        # normal tab text colour
TAB_TEXT_ACTIVE = (0, 0, 0)             # active tab text colour (black)
TAB_TEXT_ACTIVE_HOVER = (229, 233, 240) # active tab hover text colour

# Alpha levels approximating CSS opacity settings
TAB_ALPHA_NORMAL = 180   # ~0.7 * 255
TAB_ALPHA_HOVER  = 255   # 1.0
TAB_ALPHA_ACTIVE = 230   # ~0.9 * 255

# Border outline for the tab look
TAB_BORDER_RGBA = (0, 0, 0, 40)

TAB_TRANSITION_SPEED = 12.0

# =========================================================
# Disabled button appearance (matches "Run" when cannot run)
# =========================================================
# reuse styling for "selected grid size" so it looks inactive
# and is not clickable
DISABLED_BG = (110, 110, 110)
DISABLED_ALPHA = 140
DISABLED_TEXT = UI_TEXT_COLOR

# --- Fonts (scaled) ---
FONT_PATH = 'font/NotoSansMono-VariableFont_wdth,wght.ttf'


def load_font(path, size):
    """Attempt to load a font from file; fall back to a system mono font if missing."""
    try:
        return pygame.font.Font(path, size)
    except FileNotFoundError:
        print(f"Font not found: {path}, falling back to system font")
        return pygame.font.SysFont("consolas", size)


# Create fonts at scaled sizes
FONT = load_font(FONT_PATH, max(10, S(BASE_FONT_SIZE)))
FONT_BIG = load_font(FONT_PATH, max(12, S(BASE_FONT_BIG_SIZE)))
FONT_BIG.set_bold(True)  # title


# =========================================================
# Compact button width helper (text + padding)
# =========================================================
def button_width_for(text: str) -> int:
    """
    Compute a compact button width based on rendered text width.
    This reduces horizontal empty space so all tabs can fit on one row.
    """
    pad_x = S(BASE_BTN_PAD_X)
    min_w = S(BASE_BTN_MIN_W)
    return max(min_w, FONT.size(text)[0] + 2 * pad_x)


# --- Compute UI height dynamically so text never overlaps buttons ---
TITLE_H = FONT_BIG.get_height()  # title pixel height
LINE_H = FONT.get_height()       # normal line pixel height

BTN_H = S(BASE_BTN_H)

# UI height is computed from margins + title + status lines + button row
UI_HEIGHT = (
    S(BASE_MARGIN_Y) + TITLE_H +
    S(BASE_MARGIN_Y) + (STATUS_LINE_COUNT * LINE_H) +
    S(BASE_TEXT_GAP) +
    BTN_H +
    S(BASE_MARGIN_Y)
)

# Window size = grid area (square) + UI bar underneath
# Window width stays equal to GRID_WIDTH (no empty space beside the grid).
WIDTH = GRID_WIDTH
HEIGHT = GRID_WIDTH + UI_HEIGHT

# Create pygame window
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding Algorithm")

# Global heuristic pointer (set when selecting Manhattan/Euclidean)
heuristic_func = None

# ---------- Grid axis helpers (prevents last row/col stretching) ----------
def build_axes(rows: int, grid_width: int):
    """
    Build exact pixel boundaries for each row/col.
    This avoids rounding issues that cause the last row/col to stretch.
    axis length = rows + 1, where axis[-1] == grid_width.
    """
    return [round(i * grid_width / rows) for i in range(rows + 1)]


def find_index(axis, value):
    """
    Find index i such that axis[i] <= value < axis[i+1].
    Used to convert mouse pixels into grid row/col indices.
    """
    # Linear scan is fine for <= ~100 rows and keeps dependencies minimal
    for i in range(len(axis) - 1):
        if axis[i] <= value < axis[i + 1]:
            return i
    return None


class Node:
    """Represents a single cell in the grid (a node for A*)."""
    def __init__(self, row, col, x_axis, y_axis, total_rows):
        self.row = row
        self.col = col

        # Pixel bounds derived from axis arrays (prevents stretching artifacts)
        self.x = x_axis[col]                       # left pixel
        self.y = y_axis[row]                       # top pixel
        self.w = x_axis[col + 1] - x_axis[col]     # width in pixels
        self.h = y_axis[row + 1] - y_axis[row]     # height in pixels

        # Visual state
        self.color = GRID_BG_COLOR

        # Neighbours (adjacent walkable nodes)
        self.neighbours = []

        # Needed for boundary checks in neighbour updates
        self.total_rows = total_rows

    def get_pos(self):
        """Return (row, col) for heuristics and comparisons."""
        return self.row, self.col

    def is_obstacle(self):
        """True if this node is currently blocked."""
        return self.color == OBSTACLE_COLOR

    # State setters (just colour changes)
    def reset(self):
        self.color = GRID_BG_COLOR

    def make_start(self):
        self.color = START_COLOR

    def make_closed(self):
        self.color = CLOSED_COLOR

    def make_open(self):
        self.color = OPEN_COLOR

    def make_obstacle(self):
        self.color = OBSTACLE_COLOR

    def make_end(self):
        self.color = END_COLOR

    def make_path(self):
        self.color = PATH_COLOR

    def draw(self, window):
        """Draw this node as a filled rectangle."""
        pygame.draw.rect(window, self.color, (self.x, self.y, self.w, self.h))

    def update_neighbours(self, grid):
        """
        Build a list of walkable neighbours (up/down/left/right).
        Called before running A*.
        """
        self.neighbours = []

        # Up
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():
            self.neighbours.append(grid[self.row - 1][self.col])

        # Down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_obstacle():
            self.neighbours.append(grid[self.row + 1][self.col])

        # Left
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():
            self.neighbours.append(grid[self.row][self.col - 1])

        # Right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_obstacle():
            self.neighbours.append(grid[self.row][self.col + 1])

    def __lt__(self, other):
        """
        Required because PriorityQueue may compare items if priorities tie.
        We don't want Node comparisons, so return False always.
        """
        return False


# ---------------- Heuristic functions ----------------
def manhattan(p1, p2):
    """Manhattan distance: |dx| + |dy| (best for 4-direction movement)."""
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean(p1, p2):
    """Euclidean distance: straight-line distance."""
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def reconstruct_path(came_from, current, draw):
    """
    Walk backwards using came_from links, color the path, and redraw.
    Returns number of nodes in the path (excluding start).
    """
    path_length = 0
    while current in came_from:
        path_length += 1
        current = came_from[current]
        current.make_path()
        draw()
    return path_length


def algorithm(draw, grid, start, end, heuristic):
    """
    Run A* and visualise each step via draw().
    Returns:
      (True, {"time": seconds, "path_len": n})  if found
      (False, {"time": seconds, "path_len": 0}) if no path
      (False, None) if the user closed the window mid-run
    """
    count = 0  # tie-breaker for PriorityQueue
    open_set = PriorityQueue()
    open_set.put((0, count, start))  # (f_score, tie, node)
    came_from = {}  # path reconstruction map: node -> previous node

    # Initialise scores
    g_score = {node: float("inf") for row in grid for node in row}  # cost from start
    g_score[start] = 0

    f_score = {node: float("inf") for row in grid for node in row}  # estimated total cost
    f_score[start] = heuristic(start.get_pos(), end.get_pos())

    # Track membership of open set efficiently
    open_set_hash = {start}

    start_time = timer()
    while not open_set.empty():
        # Allow quit during algorithm run
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False, None

        # Get node with lowest f_score
        current = open_set.get()[2]
        open_set_hash.remove(current)

        # Found the goal: reconstruct and return
        if current == end:
            end_time = timer()
            path_len = reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            return True, {"time": end_time - start_time, "path_len": path_len}

        # Explore neighbours
        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1  # each move costs 1
            if temp_g_score < g_score[neighbour]:
                # Found a better route to neighbour
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + heuristic(neighbour.get_pos(), end.get_pos())

                # Add neighbour to open set if not already present
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()  # colour as "open"

        # Update display
        draw()

        # Mark as explored (closed) unless it's the start node
        if current != start:
            current.make_closed()

    # If we exit the loop, no path exists
    end_time = timer()
    return False, {"time": end_time - start_time, "path_len": 0}


def make_grid(rows, grid_width):
    """
    Create a rows x rows grid of Node objects and the axis arrays used for pixel mapping.
    """
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
    """Draw grid lines over the node rectangles."""
    # Vertical lines
    for x in x_axis:
        pygame.draw.line(window, GRID_LINE_COLOR, (x, 0), (x, grid_width))
    # Horizontal lines
    for y in y_axis:
        pygame.draw.line(window, GRID_LINE_COLOR, (0, y), (grid_width, y))


def get_clicked_pos(pos, rows, grid_width, x_axis, y_axis):
    """
    Convert a mouse pixel position to a (row, col) in the grid.
    Returns None if click is in the UI bar.
    """
    x, y = pos
    if y >= grid_width:
        return None  # clicked below grid area (in UI)

    col = find_index(x_axis, x)
    row = find_index(y_axis, y)
    if row is None or col is None:
        return None
    return row, col


# =========================================================
#                    Simple UI widgets
# =========================================================

def clamp(x, lo, hi):
    """Clamp x into [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    """Linear interpolate from a to b by t (0..1)."""
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    """Linear interpolate between two RGB colours."""
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t)),
    )


class Button:
    """
    A tab-like button:
    - Has hovered and active states
    - Smoothly transitions background and opacity over time (dt-based)

    CHANGE: "enabled=False" is now used for TWO cases:
      1) normal disabled buttons (e.g. Run when cannot run)
      2) the selected grid size button (so it looks inactive AND is not clickable)
    """
    def __init__(self, rect, text, on_click, active_style=True):
        self.rect = pygame.Rect(rect)  # clickable rectangle
        self.text = text               # button label
        self.on_click = on_click       # callback when clicked

        # Pseudo-class state
        self.hovered = False
        self.active = False

        # If False, the "active" style is never shown
        self.active_style = bool(active_style)

        # Current visual state for smooth transitions
        self._alpha = TAB_ALPHA_NORMAL
        self._bg = TAB_BG_NORMAL

    def set_active(self, active: bool):
        """Set active tab state (used for heuristic selection and tracking grid selection)."""
        self.active = bool(active)

    def update(self, dt: float, mouse_pos):
        """
        Update hover state and animate toward target visuals.
        dt = seconds since last frame.
        """
        self.hovered = self.rect.collidepoint(mouse_pos)

        # Decide target colours/alpha based on visual state.
        # If active_style is False, the button never shows "active" styling.
        visually_active = self.active and self.active_style

        if visually_active:
            target_bg = TAB_BG_ACTIVE
            target_alpha = TAB_ALPHA_ACTIVE
        else:
            target_bg = TAB_BG_NORMAL
            target_alpha = TAB_ALPHA_HOVER if self.hovered else TAB_ALPHA_NORMAL

        # Interpolate toward targets (CSS-like transition)
        t = clamp(dt * TAB_TRANSITION_SPEED, 0.0, 1.0)
        self._alpha = int(lerp(self._alpha, target_alpha, t))
        self._bg = lerp_color(self._bg, target_bg, t)

    def draw(self, surf, enabled=True):
        """
        Draw the button.
        If disabled, render it greyed out and ignore hover styling.
        """
        radius = S(BASE_BTN_RADIUS)

        # Disabled styling (used for Run when cannot run and selected grid size button)
        if not enabled:
            bg = DISABLED_BG
            alpha = DISABLED_ALPHA
            text_col = DISABLED_TEXT
        else:
            bg = self._bg
            alpha = self._alpha
            visually_active = self.active and self.active_style
            if visually_active:
                text_col = TAB_TEXT_ACTIVE_HOVER if self.hovered else TAB_TEXT_ACTIVE
            else:
                text_col = TAB_TEXT_NORMAL

        # Render into a temporary surface so can apply opacity
        tmp = pygame.Surface(self.rect.size, pygame.SRCALPHA)

        # Background
        pygame.draw.rect(tmp, (*bg, 255), tmp.get_rect(), border_radius=radius)

        # Border outline
        pygame.draw.rect(tmp, TAB_BORDER_RGBA, tmp.get_rect(), S(1), border_radius=radius)

        # Apply alpha like CSS opacity
        tmp.set_alpha(alpha)
        surf.blit(tmp, self.rect.topleft)

        # Draw label centered in the button
        label = FONT.render(self.text, True, text_col)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def handle_event(self, event, enabled=True):
        """Trigger click callback on left mouse click if enabled."""
        if not enabled:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()


def draw_ui_bar(window, status_lines, buttons):
    """Draw the UI bar (title, status lines, and buttons) under the grid."""
    ui_top = GRID_WIDTH  # y coordinate where the UI bar starts
    ui_rect = pygame.Rect(0, ui_top, WIDTH, UI_HEIGHT)
    pygame.draw.rect(window, UI_BG_COLOR, ui_rect)

    mx = S(BASE_MARGIN_X)
    my = S(BASE_MARGIN_Y)

    # Title
    title = FONT_BIG.render("A* Pathfinding Visualiser", True, UI_TITLE_COLOR)
    window.blit(title, (mx, ui_top + my))

    # Status text block (limited to STATUS_LINE_COUNT lines)
    text_top = ui_top + my + TITLE_H + my
    y = text_top
    for line in status_lines[:STATUS_LINE_COUNT]:
        txt = FONT.render(line, True, UI_TEXT_COLOR)
        window.blit(txt, (mx, y))
        y += LINE_H

    # Buttons are anchored to the bottom of the UI bar
    btn_row_y = ui_top + UI_HEIGHT - my - BTN_H
    for b, enabled in buttons:
        b.rect.y = btn_row_y
        b.draw(window, enabled=enabled)


def draw_all(window, grid, rows, grid_width, status_lines, buttons, x_axis, y_axis):
    """Draw the grid, grid lines, UI bar, then flip the display."""
    # Clear only the grid area (not the whole window)
    window.fill(GRID_BG_COLOR, rect=pygame.Rect(0, 0, grid_width, grid_width))

    # Draw each node
    for row in grid:
        for node in row:
            node.draw(window)

    # Draw grid lines on top
    draw_grid(window, rows, grid_width, x_axis, y_axis)

    # Draw UI section
    draw_ui_bar(window, status_lines, buttons)

    # Present the frame
    pygame.display.update()


def main():
    global heuristic_func

    # =========================================================
    # Grid size presets (shown as extra tabs in CHOOSE_HEURISTIC)
    # =========================================================
    GRID_SIZE_OPTIONS = (30, 45, 60)  # <-- preset options

    # Selected grid size (default matches your previous constant)
    rows_selected = 45

    # Build initial grid (uses selected grid size)
    grid, x_axis, y_axis = make_grid(rows_selected, GRID_WIDTH)

    # Start and end nodes (chosen by user)
    start = None
    end = None

    # Simple state machine for interaction flow
    state = "CHOOSE_HEURISTIC"
    last_result = None  # persists across Clear; updates only when a run finishes
    status_message = "Choose a heuristic to begin."

    def set_manhattan():
        """Select Manhattan heuristic and enable editing mode."""
        nonlocal state, status_message
        globals()["heuristic_func"] = manhattan
        state = "EDITING"
        status_message = "Manhattan selected. Click to place Start/End, drag to add obstacles."

        # Visual tab state
        b1.set_active(True)
        b2.set_active(False)

    def set_euclidean():
        """Select Euclidean heuristic and enable editing mode."""
        nonlocal state, status_message
        globals()["heuristic_func"] = euclidean
        state = "EDITING"
        status_message = "Euclidean selected. Click to place Start/End, drag to add obstacles."

        # Visual tab state
        b1.set_active(False)
        b2.set_active(True)

    # =========================================================
    # Grid size button callbacks
    # =========================================================
    def set_grid_size(n: int):
        """
        Rebuild the grid at a new resolution (rows x rows).
        This is only enabled while choosing a heuristic, so users
        can decide the granularity before editing/running A*.
        """
        nonlocal grid, x_axis, y_axis, start, end, state, status_message, rows_selected

        # Store the user's selection so status + Clear keep the same size
        rows_selected = int(n)

        # Reset any placed nodes because indices/pixels change with new size
        start = None
        end = None

        # Rebuild grid + axes using the selected size
        grid, x_axis, y_axis = make_grid(rows_selected, GRID_WIDTH)

        # Keep the user in CHOOSE_HEURISTIC (size is chosen alongside heuristic)
        state = "CHOOSE_HEURISTIC"
        status_message = f"Grid size set to {rows_selected}×{rows_selected}. Choose a heuristic."

        # Track which size is selected (we will disable the selected button in buttons_with_enabled)
        b_size_30.set_active(rows_selected == 30)
        b_size_45.set_active(rows_selected == 45)
        b_size_60.set_active(rows_selected == 60)

    def clear_grid():
        """
        Reset start/end and obstacles by rebuilding the grid.
        Note: last_result is intentionally NOT cleared (only updates after a completed run).
        """
        nonlocal grid, x_axis, y_axis, start, end, state, status_message
        start = None
        end = None

        # Clear keeps the currently selected grid size (rows_selected)
        grid, x_axis, y_axis = make_grid(rows_selected, GRID_WIDTH)

        state = "CHOOSE_HEURISTIC"
        status_message = "Grid cleared. Choose a heuristic."

        # Clearing also resets the selected heuristic tab
        b1.set_active(False)
        b2.set_active(False)

        # Keep selection tracking for grid size
        b_size_30.set_active(rows_selected == 30)
        b_size_45.set_active(rows_selected == 45)
        b_size_60.set_active(rows_selected == 60)

    def run_astar():
        """
        Run A* if start/end + heuristic are set.
        Stores last_result for both success and failure.
        """
        nonlocal state, last_result, status_message

        # Guard: must have start, end, and chosen heuristic
        if not (start and end and globals()["heuristic_func"]):
            return

        # Precompute neighbour lists for all nodes
        for row in grid:
            for node in row:
                node.update_neighbours(grid)

        # Update UI state while running
        state = "RUNNING"
        status_message = "Running A*..."

        # Run the algorithm, providing a draw callback so it can animate
        found, info = algorithm(
            lambda: draw_all(WINDOW, grid, rows_selected, GRID_WIDTH, status_lines(), buttons_with_enabled(), x_axis, y_axis),
            grid, start, end, globals()["heuristic_func"]
        )

        # If the user closed the window mid-run, algorithm returns (False, None)
        if info is None:
            return

        # Record result for BOTH success and failure (updates only when run finishes)
        last_result = {
            "found": bool(found),
            "path_len": int(info.get("path_len", 0)),
            "time": info.get("time", None),
        }

        state = "DONE"

        # Format time nicely for display
        t = last_result["time"]
        if isinstance(t, (int, float)):
            t_str = f"{t:.4f}s"
        else:
            t_str = "N/A"

        # Status message depends on success/failure
        if found:
            status_message = f"Done. Path found: {last_result['path_len']} nodes, {t_str}"
        else:
            status_message = f"Done. No path found. Time: {t_str}"

    # ---- Create buttons (positioned later in draw_ui_bar) ----
    mx = S(BASE_MARGIN_X)
    gap = S(BASE_BTN_GAP)

    # =========================================================
    # Compact button widths based on text + padding
    # =========================================================
    # This reduces horizontal empty space either side of the label, so all
    # buttons are more likely to fit on one row without widening the window.
    w1 = button_width_for("Manhattan")
    w2 = button_width_for("Euclidean")
    w3 = button_width_for("Run")
    w4 = button_width_for("Clear")

    x = mx
    b1 = Button((x, 0, w1, BTN_H), "Manhattan", set_manhattan, active_style=True)
    x += w1 + gap
    b2 = Button((x, 0, w2, BTN_H), "Euclidean", set_euclidean, active_style=True)
    x += w2 + gap
    b3 = Button((x, 0, w3, BTN_H), "Run", run_astar, active_style=False)
    x += w3 + gap
    b4 = Button((x, 0, w4, BTN_H), "Clear", clear_grid, active_style=False)
    x += w4 + gap

    # =========================================================
    # Grid size tabs (same Button class / same styling)
    # =========================================================
    # These are only enabled during CHOOSE_HEURISTIC, just like the heuristic tabs.
    # Labels are short to keep the button row compact at smaller SCALE values.
    size_w = button_width_for("60")  # compact width based on text + padding

    b_size_30 = Button((x, 0, size_w, BTN_H), "30", lambda: set_grid_size(30), active_style=False)
    x += size_w + gap
    b_size_45 = Button((x, 0, size_w, BTN_H), "45", lambda: set_grid_size(45), active_style=False)
    x += size_w + gap
    b_size_60 = Button((x, 0, size_w, BTN_H), "60", lambda: set_grid_size(60), active_style=False)

    # Track initial selection
    b_size_30.set_active(rows_selected == 30)
    b_size_45.set_active(rows_selected == 45)
    b_size_60.set_active(rows_selected == 60)

    def status_lines():
        """Build the text lines shown in the UI bar."""
        heuristic_name = (
            "Manhattan" if globals()["heuristic_func"] == manhattan else
            "Euclidean" if globals()["heuristic_func"] == euclidean else
            "None"
        )
        lines = [
            f"Mode: {state}    Heuristic: {heuristic_name}    Grid: {rows_selected}x{rows_selected}",  # show grid size
            status_message,
            "Left-click: place Start then End. Drag: obstacles. Right-click: erase.",
        ]

        # Show last run metrics if available (persists across clears)
        if last_result:
            t = last_result["time"]
            t_str = f"{t:.4f}s" if isinstance(t, (int, float)) else "N/A"
            lines.append(
                f"Last run: found={last_result['found']} path_len={last_result['path_len']} time={t_str}"
            )
        return lines

    def buttons_with_enabled():
        """
        Return list of (button, enabled_bool) based on current state.

        CHANGE:
        - Grid size buttons are enabled only during CHOOSE_HEURISTIC
        - The CURRENTLY SELECTED grid size button is DISABLED (enabled=False)
          so it looks greyed out and is not clickable, matching "Run" when it cannot run.
        """
        choose = (state == "CHOOSE_HEURISTIC")
        can_run = (state in ("EDITING", "DONE")) and start and end and globals()["heuristic_func"]

        # disable the selected grid size button (so it is not clickable)
        size_30_enabled = choose and (rows_selected != 30)
        size_45_enabled = choose and (rows_selected != 45)
        size_60_enabled = choose and (rows_selected != 60)

        return [
            (b1, choose),             # heuristic buttons only enabled when choosing
            (b2, choose),
            (b3, bool(can_run)),      # run enabled only when we can run
            (b4, True),               # clear always enabled
            (b_size_30, size_30_enabled),  # selected size is disabled (looks inactive)
            (b_size_45, size_45_enabled),
            (b_size_60, size_60_enabled),
        ]

    running = True
    mouse_down_left = False  # used to support "drag to paint obstacles"

    clock = pygame.time.Clock()  # used for dt and to cap FPS

    while running:
        # dt is used for smooth UI transitions (button fade/colour interpolation)
        dt = clock.tick(60) / 1000.0  # seconds since last frame (60 FPS cap)
        mouse_pos = pygame.mouse.get_pos()

        # Update button hover/transition states every frame
        for btn, _enabled in buttons_with_enabled():
            btn.update(dt, mouse_pos)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Let buttons react to clicks
            for btn, enabled in buttons_with_enabled():
                btn.handle_event(event, enabled=enabled)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down_left = True  # begin dragging/placing
                elif event.button == 3:
                    # Right click erases (only in editing/done)
                    pos = get_clicked_pos(event.pos, rows_selected, GRID_WIDTH, x_axis, y_axis)
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

            # Keyboard shortcuts
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    run_astar()
                if event.key == pygame.K_c:
                    clear_grid()

        # While holding left mouse, place start/end/obstacles (only in editing/done)
        if state in ("EDITING", "DONE") and mouse_down_left:
            mpos = pygame.mouse.get_pos()
            pos = get_clicked_pos(mpos, rows_selected, GRID_WIDTH, x_axis, y_axis)
            if pos is not None:
                r, c = pos
                node = grid[r][c]

                # First click sets start, second sets end, then drag paints obstacles
                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != start and node != end:
                    node.make_obstacle()

        # Draw the current frame
        draw_all(WINDOW, grid, rows_selected, GRID_WIDTH, status_lines(), buttons_with_enabled(), x_axis, y_axis)

    pygame.quit()  # clean shutdown


if __name__ == "__main__":
    main()
