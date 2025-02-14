from Environment.GridMap import GridMap


class Map_1(GridMap):
    def __init__(self):
        super().__init__(40, 40, 0.01)
        # Walls
        self.add_obstacle(0, 0, 1, self.height)
        self.add_obstacle(self.width-1, 0, 1, self.height)
        self.add_obstacle(0, 0, self.width,1)
        self.add_obstacle(0, self.height-1, self.width,1)

        # Obstacles (maze-like pattern)
        self.add_obstacle(5, 5, 1, 10)
        self.add_obstacle(5, 15, 10, 1)

        self.add_obstacle(10, 25, 1, 10)
        self.add_obstacle(5, 25, 10, 1)

        self.add_obstacle(20, 5, 1, 10)
        self.add_obstacle(25, 15, 10, 1)

        self.add_obstacle(35, 5, 1, 10)
        self.add_obstacle(30, 10, 10, 1)

        self.add_obstacle(20, 20, 1, 10)
        self.add_obstacle(35, 20, 1, 10)

        self.add_obstacle(25, 25, 10, 1)
        self.add_obstacle(30, 25, 10, 1)

        self.add_obstacle(17,35,15,1)