class Pose:
    def __init__(self, points):
        assert len(points) == 21, "Not enough points!"

        self.right_eye = 1
        self.left_eye = 1
        self.nose = 1
        self.left_p =