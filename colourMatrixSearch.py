"""Find the longest connection of same colours in the matrix."""
import numpy as np


def get_neighbours(visited, matrix, row, col):
    """Get the adjacent neighbours horizontally and vertically."""
    neighbours = []

    if col + 1 < len(matrix):
        if not_visited(visited, row, col + 1):
            neighbours.append({"val": matrix[row][col + 1], "dir": "R"})

    if col - 1 >= 0:
        if not_visited(visited, row, col - 1):
            neighbours.append({"val": matrix[row][col - 1], "dir": "L"})

    if row - 1 >= 0:
        if not_visited(visited, row - 1, col):
            neighbours.append({"val": matrix[row - 1][col], "dir": "U"})

    if row + 1 < len(matrix):
        if not_visited(visited, row + 1, col):
            neighbours.append({"val": matrix[row + 1][col], "dir": "D"})

    return neighbours


def not_visited(visited, row, col):
    """Check if visited."""
    for v in visited:
        if v['row'] == row and v['col'] == col:
            return False

    return True


def same_colour(neighbour, row, col):
    """Check if the neighbour is the same colour."""
    return matrix[row][col] == neighbour['val']


def new_row_and_col(neighbour, row, col):
    """Update the row and col based on the same colour neighbour location."""
    if neighbour["dir"] == "R":
        return row, col + 1

    if neighbour["dir"] == "L":
        return row, col - 1

    if neighbour["dir"] == "U":
        return row - 1, col

    if neighbour["dir"] == "D":
        return row + 1, col


def depth_first_search(visited, matrix, row, col, longest):
    """Perform depth first search."""
    visited.append({'row': row, 'col': col})
    longest = longest + 1
    neighbours = get_neighbours(visited, matrix, row, col)
    for neighbour in neighbours:
        if same_colour(neighbour, row, col):
            row, col = new_row_and_col(neighbour, row, col)
            longest = depth_first_search(visited, matrix, row, col, longest)

    return longest

current_longest = 0
max_longest = 0

matrix = np.array([
    [1, 2, 2],
    [3, 3, 3],
    [3, 2, 1]])

visited = []

for row in range(len(matrix)):
    for col in range(len(matrix)):
        current_longest = depth_first_search(visited, matrix, row, col, current_longest)
        if current_longest > max_longest:
            max_longest = current_longest

        current_longest = 0

print(max_longest)
