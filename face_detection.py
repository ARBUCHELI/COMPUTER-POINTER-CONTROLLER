import cv2
import numpy as np
from model import Model

class FaceDetection(Model):
    def __init__(self, model_name, device='CPU', extensions=None):
        
        Model.__init__(self, model_name, device, extensions)
        self.model_name = model_name
        self.input_blob = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.out_blob = next(iter(self.net.outputs))
        self.out_shape = self.net.outputs[self.out_blob].shape
        
    def predict(self, image, p_th):
        predicted_img = self.preprocess_input(image)
        input_dict = {self.input_blob:predicted_img}
        outputs = self.net_plugin.infer(input_dict)
        coords = self.preprocess_output(outputs, p_th)
        if (len(coords)==0):
            return 0, 0
        coords = coords[0] 
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32) 
        face_retrieved = image[coords[1]:coords[3], coords[0]:coords[2]] 
        return face_retrieved, coords
        
    def check_model(self):
        '''
        Already implemented in the load_model method.
        '''
        
    def preprocess_input(self, image):

        preprocessed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        preprocessed_image = np.transpose(np.expand_dims(preprocessed_image,axis=0), (0,3,1,2))

        return preprocessed_image

    def preprocess_output(self, outputs, p_th):

        coords = []
        sals = outputs[self.out_blob][0][0]
        for box in sals:
            if box[2]>p_th:
                x_min=box[3]
                y_min=box[4]
                x_max=box[5]
                y_max=box[6]
                coords.append([x_min,y_min,x_max,y_max])
        
        return coords
                
    

      


