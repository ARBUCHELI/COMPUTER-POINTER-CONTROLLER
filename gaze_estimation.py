import cv2
import numpy as np
import math
from model import Model

class GazeEstimation(Model):
    def __init__(self, model_name, device='CPU', extensions=None):
        
        Model.__init__(self, model_name, device, extensions)
        self.model_name = model_name
        self.input_blob = [i for i in self.net.inputs.keys()]
        self.input_shape = self.net.inputs[self.input_blob[1]].shape
        self.out_blob = [i for i in self.net.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        img_left_predicted, img_right_predicted = self.preprocess_input(left_eye_image, right_eye_image)
        input_dict = {'head_pose_angles':head_pose_angles, 'left_eye_image':img_left_predicted, 'right_eye_image':img_right_predicted}
        outputs = self.net_plugin.infer(input_dict)
        coordi_update_pointer, coordi_gaze = self.preprocess_output(outputs, head_pose_angles)

        return coordi_update_pointer, coordi_gaze

    def check_model(self):
        '''
        Already implemented in load_model method.
        '''

    def preprocess_input(self, image_lefte, image_righte):
        image_left_process = cv2.resize(image_lefte, (self.input_shape[3], self.input_shape[2]))
        image_right_process = cv2.resize(image_righte, (self.input_shape[3], self.input_shape[2]))
        img_left_processed = np.transpose(np.expand_dims(image_left_process,axis=0), (0,3,1,2))
        img_right_processed = np.transpose(np.expand_dims(image_right_process,axis=0), (0,3,1,2))
        return img_left_processed, img_right_processed

    def preprocess_output(self, outputs, gaze_angles):
        coordi_gaze = outputs[self.out_blob[0]].tolist()[0]
        gaze = gaze_angles[2] 
        abscissaVal = math.cos(gaze * math.pi / 180.0)
        ordinateVal = math.sin(gaze * math.pi / 180.0)
        
        xupdated = coordi_gaze[0] * abscissaVal + coordi_gaze[1] * ordinateVal
        yupdated = coordi_gaze[0] *ordinateVal + coordi_gaze[1] * abscissaVal
        return (xupdated,yupdated), coordi_gaze
