import cv2
import numpy as np
from model import Model

class FacialLandmarksDetection(Model):
    def __init__(self, model_name, device='CPU', extensions=None):

        Model.__init__(self, model_name, device, extensions)
        self.model_name = model_name
        self.input_blob = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.out_blob = next(iter(self.net.outputs))
        self.out_shape = self.net.outputs[self.out_blob].shape

    def predict(self, image):
        predicted_image = self.preprocess_input(image)
        input_dict = {self.input_blob: predicted_image}
        outputs = self.net_plugin.infer(input_dict)
        coords = self.preprocess_output(outputs)
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32) 

        xmin_left_eye = coords[0]-10
        ymin_left_eye = coords[1]-10
        xmax_left_eye = coords[0]+10
        ymax_left_eye = coords[1]+10
        
        xmin_right_eye = coords[2]-10
        ymin_right_eye = coords[3]-10
        xmax_right_eye = coords[2]+10
        ymax_right_eye = coords[3]+10

        eye_left_detect =  image[ymin_left_eye:ymax_left_eye, xmin_left_eye:xmax_left_eye]
        eye_right_detect = image[ymin_right_eye:ymax_right_eye, xmin_right_eye:xmax_right_eye]
        eye_coordinates_detect = [[xmin_left_eye,ymin_left_eye,xmax_left_eye,ymax_left_eye], [xmin_right_eye,ymin_right_eye,xmax_right_eye,ymax_right_eye]]

        return eye_left_detect, eye_right_detect, eye_coordinates_detect
        
    def check_model(self):
        '''
        Already implemented in the load_model method.
        '''

    def preprocess_input(self, image):
        image_first_process = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_process = cv2.resize(image_first_process, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_process, axis=0), (0,3,1,2))
        return img_processed

    def preprocess_output(self, outputs):
        landmarks = outputs[self.out_blob][0]
        xlefteye = landmarks[0].tolist()[0][0]
        ylefteye = landmarks[1].tolist()[0][0]
        xrighteye = landmarks[2].tolist()[0][0]
        yrighteye = landmarks[3].tolist()[0][0]
        
        return (xlefteye, ylefteye, xrighteye, yrighteye)

