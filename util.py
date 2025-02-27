from __future__ import annotations
from typing import NamedTuple
from Maze import Action

class Coord(NamedTuple):
    """X,Y Maze Coordinates"""
    x: int
    y: int

    def __repr__(self) -> str: return repr(tuple(self))
    def __str__(self) -> str: return str(tuple(self))


class SearchResult(NamedTuple):
    """The return value for search methods.
    reachable: Whether the goal is reachable from the start.
    path: The path from the start of the maze to the end of the maze.
          If no path exists, then None.
    expanded_list: The list of expanded node coordinates.
    """
    reachable: bool
    path: tuple[list[Coord], list[Action]] | None
    expanded_list: list[Coord]

    def __repr__(self) -> str: return repr(tuple(self))
    def __str__(self) -> str: return str(tuple(self))
