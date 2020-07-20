import cv2
import json

import tensorflow as tf

from src.PoseExtractor import PoseExtractor
from src.data_processing.pose_utils import *
from tensorflow.keras.models import model_from_json

import os



def get_biggest_pose(poses):
    max_distance = -1
    biggest_pose = None
    if len(poses):
        for pose in poses:
            pose_key_point1 = pose.points[1]
            pose_key_point2 = pose.points[8]
            if all(pose_key_point1.int_xy) and all(pose_key_point2.int_xy):
                distance = get_distance_between_points(pose_key_point1, pose_key_point2)
                if distance >= max_distance:
                    max_distance = distance
                    biggest_pose = pose
        return biggest_pose
    
    
def find_distance(pt1, pt2):
    return np.sqrt((pt2.x - pt1.x)**2 + (pt2.y - pt1.y)**2)

    
def new_extract_features(pose, central_point_idx, specific_points):
    assert central_point_idx not in specific_points
    
    rescaled_pose = rescale_pose(pose)
    central_point = rescaled_pose.points[central_point_idx]
    
    features = []
    
    for current_points_idx in specific_points:
        current_point = rescaled_pose.points[current_points_idx]

        dx = central_point.x - current_point.x
        dy = central_point.y - current_point.y
    
        features.append(dx)
        features.append(dy)
    
    return np.array(features)


class FeaturesSequence():
    
    default_pose = 0
    
    def __init__(self, max_features_num):
        self.__max_features_num = max_features_num
        self.__features_sequence = [0] * self.__max_features_num
    
    def update_features_sequense(self, new_features):
        if len(self.__features_sequence) >= self.__max_features_num:
            self.__features_sequence = self.__features_sequence[1:]
        self.__features_sequence.append(new_features)
        
    def get_vectorized_features(self):
        return self.__features_to_vec(self.__features_sequence).reshape(1, -1)
        
    @staticmethod
    def __features_to_vec(features):
        vectors = []
        for f in features:
            zeros = np.zeros((6))
            zeros[f-1] = 1
            vectors.append(zeros)
        return np.array(vectors)

    
capture = cv2.VideoCapture(0)
pose_extractor = PoseExtractor()
top_pose_points = [0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]

# features_seq = FeaturesSequence(58)





#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

labels_dict = {"hands_down": 0,
               "stop": 1,
               "hands_up": 2,
               "hands_up_small": 3,
               "hands_down_small": 4,
               "hads_down_up": 5,
               "hands_to_sides": 6}

decode_labels_dict = {val: key for key, val in labels_dict.items()}


dynamic_gestures_labesl_dict = {0: "stop",
                                1: "come_on",
                                2: "go_away",
                                3: "look_at_me",
                                4: "other"}

gestures_to_num_dict = dict(zip(dynamic_gestures_labesl_dict.values(), dynamic_gestures_labesl_dict.keys()))


def save_model(model, name):
    model_json = model.to_json()
    with open("{}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("{}.h5".format(name))
    print("Model {} saved!".format(name))
    
def load_model(name):
    json_file = open('{}.json'.format(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("{}.h5".format(name))
    return loaded_model

#static_model = load_model("static_model")
dynamic_model = load_model("main_model")


class FeaturesSeq():
    def __init__(self, max_len):
        self.max_len = max_len
        self.seq = []

    def update(self, new_features):
        if len(self.seq) + len(new_features) > self.max_len:
            d = (len(self.seq) + len(new_features)) - self.max_len
            self.seq = self.seq[d:]
        for feature in new_features:
            self.seq.append(feature)


f_seq = FeaturesSeq(600)

while True:
    ret, img = capture.read()
    if not ret:
        break

    img[:25, :210] = (255, 255, 255)
    poses = pose_extractor.extract_poses_from_image(img)
    actual_pose =  get_biggest_pose(poses)
    if actual_pose is not None:
        features = new_extract_features(actual_pose, 1, top_pose_points)
        #gesture = static_model.predict(features.reshape(1, -1))
        #gesture = np.argmax(gesture)
        #str_gesture = decode_labels_dict[gesture]
        # print(features)
        # features_seq.update_features_sequense(features)
        # for i in range(6):
        f_seq.update(features)
        # print(len(f_seq.seq))

        # vec = features_seq.get_vectorized_features()
        input_data = np.array(f_seq.seq)
        input_data = np.array([input_data.reshape(-1, 1)])
        dynamic_gesture_v = dynamic_model.predict(input_data)
        dynamic_gesture = np.argmax(dynamic_gesture_v)
        if dynamic_gesture == 1:
            if dynamic_gesture_v[0][1] < 0.85:
                dynamic_gesture = 4
        else:
            if dynamic_gesture_v[0][dynamic_gesture] < 0.95:
                dynamic_gesture = 4
            # print(dynamic_gesture_v[0][1], dynamic_gesture_v[0][0])

        str_dynamic_gesture = dynamic_gestures_labesl_dict[dynamic_gesture]

        cv2.putText(img, str_dynamic_gesture, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


        for point in actual_pose.points:
            center = (int(point.x), int(point.y))
            cv2.circle(img, center, 3, (0, 255, 255), -1)
            
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    k = cv2.waitKey(10)
    if k & 0xFF == 27:
        break
    
    
capture.release()
cv2.destroyAllWindows()
