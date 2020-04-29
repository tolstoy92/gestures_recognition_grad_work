class Pose:
    def __init__(self, points):
        assert len(points) == 25, "Not enough points!"
        self.__points = points

        self.__nose = points[0]
        self.__neck = points[1]
        self.__right_shoulder = points[2]
        self.__right_elbow = points[3]
        self.__right_wrist = points[4]
        self.__left_shoulder = points[5]
        self.__left_elbow = points[6]
        self.__left_wrist = points[7]
        self.__mid_hip = points[8]
        self.__right_hip = points[9]
        self.__right_knee = points[10]
        self.__right_ankle = points[11]
        self.__left_hip = points[12]
        self.__left_knee = points[13]
        self.__left_ankle = points[14]
        self.__right_eye = points[15]
        self.__left_eye = points[16]
        self.__right_ear = points[17]
        self.__left_ear = points[18]
        self.__left_big_toe = points[19]
        self.__left_small_toe = points[20]
        self.__left_heel = points[21]
        self.__right_big_toe = points[22]
        self.__right_small_toe = points[23]
        self.__right_heel = points[24]

    @property
    def points(self):
        return self.__points

    @property
    def nose(self):
        return self.__nose

    @property
    def neck(self):
        return self.__neck

    @property
    def right_shoulder(self):
        return self.__right_shoulder

    @property
    def right_elbow(self):
        return self.__right_elbow

    @property
    def right_wrist(self):
        return self.__right_wrist

    @property
    def left_shoulder(self):
        return self.__left_shoulder

    @property
    def left_elbow(self):
        return self.__left_elbow

    @property
    def left_wrist(self):
        return self.__left_wrist

    @property
    def mid_hip(self):
        return self.__mid_hip

    @property
    def right_hip(self):
        return self.__right_hip

    @property
    def right_knee(self):
        return self.__right_knee

    @property
    def right_ankle(self):
        return self.__right_ankle

    @property
    def left_hip(self):
        return self.__left_hip

    @property
    def left_knee(self):
        return self.__left_knee

    @property
    def left_ankle(self):
        return self.__left_ankle

    @property
    def right_eye(self):
        return self.__right_eye

    @property
    def left_eye(self):
        return self.__left_eye

    @property
    def right_ear(self):
        return self.__right_ear

    @property
    def left_ear(self):
        return self.__left_ear

    @property
    def left_big_toe(self):
        return self.__left_big_toe

    @property
    def left_small_toe(self):
        return self.__left_small_toe

    @property
    def left_heel(self):
        return self.__left_heel

    @property
    def right_big_toe(self):
        return self.__right_big_toe

    @property
    def right_small_toe (self):
        return self.__right_small_toe

    @property
    def right_heel(self):
        return self.__right_heel
