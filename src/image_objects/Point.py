class Point:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def xy(self):
        return self.__x, self.__y

    @property
    def int_xy(self):
        return int(round(self.__x)), int(round(self.__y))

    def __repr__(self):
        return "x: {}, y: {}".format(self.x, self.y)
