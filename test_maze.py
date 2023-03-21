from random import randint
from queue import PriorityQueue
from PIL import Image

import pygame as pg
import pygame_menu as pgm
import os
import glob


WIDTH, HEIGHT = 1000, 700


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
VISIT = (1, 106, 106)
PATH = (255, 140, 0)
RED = (204, 0, 0)
BLUE = (0, 0, 204)
GREEN = (0, 204, 0)


class Grid:
    def __init__(self, rows, cols, tile_size, offset):
        self.width = cols  # + (1 if cols % 2 == 0 else 2)
        self.height = rows  # + (1 if rows % 2 == 0 else 2)
        self.offset = offset
        self.tile_size = tile_size
        self.maze = [[Cell(i, j, tile_size, rows, cols, offset) for i in range(self.width)] for j in range(self.height)]

    def draw(self, screen, border):
        for i in range(self.height):
            for j in range(self.width):
                self.maze[i][j].draw(screen, border)

    def update_neighbors(self):
        for row in self.maze:
            for cell in row:
                cell.update_neighbors(self)

    def get_tile(self):
        mx, my = pg.mouse.get_pos()
        mx -= self.offset[0]
        my -= self.offset[1]
        mx //= self.tile_size
        my //= self.tile_size
        if mx >= self.width or mx < 0 or my >= self.height or my < 0:
            return

        return self.maze[int(my)][int(mx)]

    def reset_colors(self):
        for row in self.maze:
            for cell in row:
                cell.reset_color()

    def increase_tile_size(self, value):
        self.tile_size += value
        for row in self.maze:
            for cell in row:
                cell.width = self.tile_size
                cell.x = cell.col * cell.width
                cell.y = cell.row * cell.width

    def increase_offset(self, value):
        self.offset[0] += value[0]
        self.offset[1] += value[1]
        for row in self.maze:
            for cell in row:
                cell.offset[0] += value[0]
                cell.offset[1] += value[1]

    def save_img(self, filename):
        rgb_list = [WHITE if cell.block else BLACK for row in self.maze for cell in row]

        img = Image.new('RGB', (self.width, self.height))
        img.putdata(rgb_list)
        img.save(f'{filename}.png')

    def save_txt(self, filename):
        with open(f'{filename}.txt', 'w') as file:
            for row in self.maze:
                for cell in row:
                    file.write('#' if cell.block else '.')
                file.write('\n')

    def load_txt(self, filename):
        with open(f'{filename}.txt', 'r') as file:
            col_len = len(file.readline()) - 1
            file.seek(0, 0)
            row_len = 0
            while file.readline():
                row_len += 1
            file.seek(0, 0)
            self.width = col_len
            self.height = row_len
            self.offset = [0, 0]
            self.maze = [[Cell(i, j, self.tile_size, self.height, self.width, self.offset) for i in range(self.width + 1)] for j in range(self.height)]

            for row in self.maze:
                for cell in row:
                    char = file.read(1)
                    if char == '\n' or '':
                        continue
                    cell.block = True if char == '#' else False

    def load_png(self, filename):
        img = Image.open(f'{filename}')
        img = img.convert('1')
        self.offset = [0, 0]
        self.width, self.height = img.size
        self.maze = [[Cell(i, j, self.tile_size, self.height, self.width, self.offset) for i in range(self.width + 1)] for j in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                if img.getpixel((j, i)) == 0:
                    self.maze[i][j].block = False
                else:
                    self.maze[i][j].block = True

        # img.save('puk.png')


class Cell:
    def __init__(self, col, row, width, row_total, col_total, offset):
        self.row = row
        self.col = col
        self.row_total = row_total
        self.col_total = col_total
        self.x = col * width
        self.y = row * width
        self.neighbors = []
        self.width = width
        if self.row % 2 == 0 and self.col % 2 == 0:
            self._block = False
            self.color = BLACK
        else:
            self._block = True
            self.color = WHITE

        self.offset = offset

    def draw(self, screen, border):
        pg.draw.rect(screen, self.color,
                     (self.x + self.offset[0], self.y + self.offset[1], self.width, self.width))
        if border:
            pg.draw.rect(screen, RED,
                         (self.x + self.offset[0], self.y + self.offset[1], self.width, self.width),
                         width=1)

    def update_neighbors(self, grid: Grid):
        r, c = self.row, self.col
        if r < self.row_total - 1 and not grid.maze[r + 1][c].block:
            self.neighbors.append(grid.maze[r + 1][c])

        if r > 0 and not grid.maze[r - 1][c].block:
            self.neighbors.append(grid.maze[r - 1][c])

        if c < self.col_total - 1 and not grid.maze[r][c + 1].block:
            self.neighbors.append(grid.maze[r][c + 1])

        if c > 0 and not grid.maze[r][c - 1].block:
            self.neighbors.append(grid.maze[r][c - 1])

    def reset_color(self):
        if self.block:
            self.color = WHITE
        else:
            self.color = BLACK

    def make_visit(self):
        self.color = VISIT

    def make_path(self):
        self.color = PATH

    @property
    def position(self):
        return self.row, self.col

    @property
    def block(self):
        return self._block

    @block.setter
    def block(self, value):
        if value:
            self.color = WHITE
        else:
            self.color = BLACK

        self._block = value

    def __lt__(self, other):
        return False


def sidewinder(grid: Grid):
    delta = 0

    if grid.width % 2 == 0:
        width = grid.width - 1
    else:
        width = grid.width

    for y in range(0, grid.height, 2):
        for x in range(0, grid.width, 2):
            if y != 0:
                if randint(0, 1) == 0 and x != width - 1:
                    grid.maze[y][x + 1].block = False
                else:
                    random_x_cell = randint(*sorted([delta, x]))

                    if random_x_cell % 2 != 0:
                        random_x_cell += 1

                    if random_x_cell > width - 1:
                        random_x_cell -= 1

                    grid.maze[y - 1][random_x_cell].block = False

                    if x != width - 1:
                        delta = x + 1
                    else:
                        delta = 0
            else:
                if x != width - 1:
                    grid.maze[y][x + 1].block = False

            yield grid


def h_function(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def visit_animation(visited):
    for cell in visited:
        r, g, b = cell.color
        if g < 150:
            g += 1
        cell.color = (r, g, b)


def reconstruct_path(came_from: dict, current):
    path = []
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)

        yield path

    # return path


def a_star(screen, start: Cell, end: Cell, h, grid, border):
    open_set = PriorityQueue()
    open_set.put((0, start))
    open_set_hash = {start}

    came_from = {}
    g_score = {node: float('inf') for row in grid.maze for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid.maze for node in row}
    f_score[start] = h(start.position, end.position)

    visited = []

    while not open_set.empty():
        current: Cell = open_set.get()[1]

        current.update_neighbors(grid)
        if current == end:
            return reconstruct_path(came_from, current)

        open_set_hash.remove(current)
        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + 1
            tentative_f_score = tentative_g_score + h(neighbor.position, end.position)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_f_score
                if neighbor not in open_set_hash:
                    open_set.put((f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

        if current != start:
            visited.append(current)
            current.make_visit()

        visit_animation(visited)
        grid.draw(screen, border)
        pg.display.update()


def draw_path(path):
    if not path:
        return

    try:
        for cell in next(path):
            cell.make_path()
    except StopIteration:
        pass


def draw_maze(grid, gen):
    try:
        grid = next(gen)
    except StopIteration:
        pass
    return grid


def str_to_num(string):
    try:
        num = int(string)
        return num
    except ValueError:
        return 1


def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    menu_surface = pg.Surface((WIDTH // 3, HEIGHT))

    maze = Grid(11, 11, 10, [0, 0])
    sw = sidewinder(maze)
    start, end = None, None
    path = None

    ###########
    menu = pgm.Menu('MAZE', menu_surface.get_width(), menu_surface.get_height(),
                            theme=pgm.themes.THEME_DARK, position=(0, 0, True))

    menu.add.label('Размер\nлабиринта')
    x_input = menu.add.text_input('X :', default='21')
    y_input = menu.add.text_input('Y :', default='21')
    gen_button = menu.add.button('Сгенерировать')
    menu.add.label('-------------------')
    menu.add.label('Выберите\nфайл')

    # txt - False, png - True
    toggle = menu.add.toggle_switch('Png|Txt', state_text=('txt', 'png'), state_color=('darkgreen', 'darkorange'))
    save_load_filename_input = menu.add.text_input('Filename: ', default='aboba')
    save_button = menu.add.button('Сохранить')
    load_button = menu.add.button('Загрузить')
    menu.add.label('-------------------')
    border = menu.add.toggle_switch('Границы', default=1)
    reset = menu.add.button('Очистить лабиринт')
    rec = menu.add.toggle_switch('GIF')
    ###########

    start_gif = False
    counter = 0

    delay = 1
    timer = delay
    run = True
    while run:
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    if not start:
                        start = maze.get_tile()
                        if start and not start.block:
                            start.color = GREEN
                            continue

                    if start and not end:
                        end = maze.get_tile()
                        if end and not end.block:
                            end.color = RED
                            path = a_star(screen, start, end, h_function, maze, border.get_value())

                if event.button == 3:
                    pass

            if event.type == pg.MOUSEWHEEL:
                maze.increase_tile_size(event.y)

            if event.type == pg.MOUSEMOTION:
                if pg.mouse.get_pressed()[1]:
                    maze.increase_offset((event.rel[0] / 1800,
                                          event.rel[1] / 1800))

            if event.type == pg.KEYUP:
                if event.key == pg.K_RETURN:
                    if gen_button.is_selected():
                        maze = Grid(str_to_num(y_input.get_value()),
                                    str_to_num(x_input.get_value()),
                                    10, [0, 0])
                        sw = sidewinder(maze)

                    # save png
                    if toggle.get_value() and save_button.is_selected():
                        maze.save_img(save_load_filename_input.get_value())

                    # save txt
                    if not toggle.get_value() and save_button.is_selected():
                        maze.save_txt(save_load_filename_input.get_value())

                    # load txt
                    if not toggle.get_value() and load_button.is_selected():
                        maze.load_txt(save_load_filename_input.get_value())
                        start, end, path = None, None, None

                    # load png
                    if toggle.get_value() and load_button.is_selected():
                        maze.load_png(save_load_filename_input.get_value())
                        start, end, path = None, None, None

                    if reset.is_selected():
                        maze.reset_colors()
                        start, end, path = None, None, None

                    if rec.is_selected():
                        start_gif = rec.get_value()
                        if not start_gif:
                            frames = [Image.open(f'dist/img{i}.png') for i in range(counter)]
                            frames[0].save(
                                f'{save_load_filename_input.get_value()}.gif',
                                save_all=True,
                                append_images=frames[1:],
                                optimize=True,
                                duration=20,
                                loop=0,
                            )
                            counter = 0
                            files = glob.glob('dist/*')
                            for f in files:
                                os.remove(f)

        screen.fill(pg.Color('black'))

        maze.draw(screen, border.get_value())

        menu.update(events)
        menu.draw(menu_surface)

        if start_gif:
            pg.image.save(screen, f'dist/img{counter}.png')
            counter += 1

        screen.blit(menu_surface, (WIDTH - menu_surface.get_width(), 0))

        if timer > 0:
            timer -= 1
        else:
            maze = draw_maze(maze, sw)
            draw_path(path)
            timer = delay

        pg.display.update()


if __name__ == '__main__':
    main()
