import cv2
import numpy as np
from model import Model

class HeadPoseEstimation(Model):
    def __init__(self, model_name, device='CPU', extensions=None):

        Model.__init__(self, model_name, device, extensions)
        self.model_name = model_name
        self.input_blob = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.out_blob = next(iter(self.net.outputs))
        self.out_shape = self.net.outputs[self.out_blob].shape

    def predict(self, image):
        predicted_img = self.preprocess_input(image)
        input_dict = {self.input_blob:predicted_img}
        outputs = self.net_plugin.infer(input_dict)
        pose_estim = self.preprocess_output(outputs)
        return pose_estim

    def check_model(self):
        '''
        Already implemented in the load_model method.
        '''

    def preprocess_input(self, image):
        
        preprocessed_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        preprocessed_img = np.transpose(np.expand_dims(preprocessed_img, axis=0), (0,3,1,2))
        return preprocessed_img

    def preprocess_output(self, outputs):
        poses = []
        poses.append(outputs['angle_y_fc'].tolist()[0][0])
        poses.append(outputs['angle_p_fc'].tolist()[0][0])
        poses.append(outputs['angle_r_fc'].tolist()[0][0])
        return poses
