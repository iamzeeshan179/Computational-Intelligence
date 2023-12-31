import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import enum


# Enum representing the directions an ant can take.
class Direction(enum.Enum):
    east = 0
    north = 1
    west = 2
    south = 3

    # Direction to an int.
    # @param dir the direction.
    # @return an integer from 0-3.
    @classmethod
    def dir_to_int(cls, dir):
        return dir.value

    @classmethod
    def int_to_dir(self, int):
        if int == 0:
            return Direction.east
        elif int == 1:
            return Direction.north
        elif int == 2:
            return Direction.west
        elif int == 3:
            return Direction.south
        else:
            return None
