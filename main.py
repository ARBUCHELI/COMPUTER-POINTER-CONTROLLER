import cv2
import os
import logging
import time
import numpy as np
from input_feeder import InputFeeder
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from argparse import ArgumentParser

def get_args():
   
    parser = ArgumentParser()
    
    dt_desc = 'The location of the face detection model XML file'
    pe_desc = 'The location of the head pose estimation model XML file'
    ld_desc = 'The location of the facial landmarks detection model XML file'
    ge_desc = 'The location of the gaze estimation model XML file'
    i_desc = 'The location of the input file'
    f_desc = 'To visualize more than one model output, enter the flags separated by space, fad for face detection, hpe for head pose estimation fld for facial landmarks detection, gae for gaze estimation'
    l_desc = 'CPU Extension for custom layers. /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
    pth_desc = 'The probability threshold to use with the bounding boxes'
    d_desc = 'The device name, CPU, GPU, FPGA, MYRIAD'
    o_desc = 'Results of all process'

    
    parser.add_argument('-dt', '--face_detection_model', required=True, type=str, help=dt_desc)
    parser.add_argument('-pe', '--head_pose_estimation_model', required=True, type=str, help=pe_desc)
    parser.add_argument('-ld', '--facial_landmarks_detection_model', required=True, type=str, help=ld_desc)
    parser.add_argument('-ge', '--gaze_estimation_model', required=True, type=str, help=ge_desc)
    parser.add_argument('-i', '--input', required=True, type=str, help=i_desc)
    parser.add_argument('-f', '-four_flags', '--flags_checker', required=False, nargs='+', default=[], help=f_desc)
    parser.add_argument('-l', '--cpu_extension', required=False, type=str, default=None, help=l_desc)
    parser.add_argument('--p_th', required=False, type=float, default=0.6, help=pth_desc)
    parser.add_argument('-d', '--device', type=str, default="CPU", help=d_desc)
    parser.add_argument("-o", '--out_path', default='/results/', type=str, help = o_desc)
    return parser


def main():

    args = get_args().parse_args()
    path_filender = args.input
    four_flags = args.flags_checker
    loger = logging.getLogger()
    feeder_in = None
    out_path = args.out_path
   
    if path_filender.lower() == "cam":
        feeder_in = InputFeeder("cam")
    else:
        if not os.path.isfile(path_filender):
            loger.error("The video was not found")
            exit(1)
        feeder_in = InputFeeder("video", path_filender)

    model_locations = {'FaceDetection':args.face_detection_model, 'HeadPoseEstimation':args.head_pose_estimation_model, 
    'FacialLandmarksDetection':args.facial_landmarks_detection_model, 'GazeEstimation':args.gaze_estimation_model}

    for key_name in model_locations.keys():
        if not os.path.isfile(model_locations[key_name]):
            loger.error("The system cannot find the " + key_name + " xml file")
            exit(1)
            
    dt = FaceDetection(model_locations['FaceDetection'], args.device, args.cpu_extension)
    pe = HeadPoseEstimation(model_locations['HeadPoseEstimation'], args.device, args.cpu_extension)
    ld = FacialLandmarksDetection(model_locations['FacialLandmarksDetection'], args.device, args.cpu_extension)
    ge = GazeEstimation(model_locations['GazeEstimation'], args.device, args.cpu_extension)
    
    
    cursor = MouseController('medium','fast')
    
    feeder_in.load_data()
    model_load_time_start = time.time()
    dt.load_model()
    pe.load_model()
    ld.load_model()
    ge.load_model()
    total_load_time = time.time() - model_load_time_start
    
    
    frame_counter = 0
    inference_time_start = time.time()
    for ret, frame in feeder_in.next_batch():
        if not ret:
            break
        frame_counter = frame_counter + 1
        if frame_counter%1==0:
            cv2.imshow('video', cv2.resize(frame, (600, 600)))
    
        key = cv2.waitKey(60)

        face_detected, coords_face = dt.predict(frame, args.p_th)
        if type(face_detected)==int:
            loger.error("The system cannot detect any face.")
            if key==27:
                break
            continue
        
        head_pose_output = pe.predict(face_detected)
        eye_left_detect, eye_right_detect, eye_coordinates_detect  = ld.predict(face_detected)
        coordi_update_pointer, coordi_gaze = ge.predict(eye_left_detect, eye_right_detect, head_pose_output)
        
        if (not len(four_flags)==0):
            result_app = frame
            if 'fad' in four_flags:
                result_app = face_detected
            if 'hpe' in four_flags:
                cv2.putText(result_app, "HP Angles: YAW:{:.3f} * PITCH:{:.3f} * ROLL:{:.3f}".format(head_pose_output[0], head_pose_output[1], head_pose_output[2]), (5, 40), cv2.FONT_HERSHEY_COMPLEX, 0.25, (153, 76, 0), 0)
            if 'fld' in four_flags:
                cv2.rectangle(face_detected, (eye_coordinates_detect[0][0]-4, eye_coordinates_detect[0][1]-4), (eye_coordinates_detect[0][2]+4, eye_coordinates_detect[0][3]+4), (255,255,0), 4)
                cv2.rectangle(face_detected, (eye_coordinates_detect[1][0]-4, eye_coordinates_detect[1][1]-4), (eye_coordinates_detect[1][2]+4, eye_coordinates_detect[1][3]+4), (255,255,0), 4)
            if 'gae' in four_flags:
                x = int(coordi_gaze[0]*2)
                y = int(coordi_gaze[1]*2)
                w = 150
                right_E = cv2.line(eye_right_detect, (x-w, y-w), (x+w, y+w), (51,255,153), 1)
                cv2.line(right_E, (x-w, y+w), (x+w, y-w), (51,255,253), 1)
                left_E = cv2.line(eye_left_detect, (x-w, y-w), (x+w, y+w), (51,255,153), 1)
                cv2.line(left_E, (x-w, y+w), (x+w, y-w), (51,255,253), 1)
                face_detected[eye_coordinates_detect[1][1]:eye_coordinates_detect[1][3],eye_coordinates_detect[1][0]:eye_coordinates_detect[1][2]] = right_E
                face_detected[eye_coordinates_detect[0][1]:eye_coordinates_detect[0][3],eye_coordinates_detect[0][0]:eye_coordinates_detect[0][2]] = left_E
               
                
            cv2.imshow("Result of the App", cv2.resize(result_app,(600, 600)))
        
        if frame_counter%5==0:
            cursor.move(coordi_update_pointer[0], coordi_update_pointer[1])    
        if key==27:
                break

    total_time = time.time() - inference_time_start
    total_time_for_inference = round(total_time, 1)
    fps = frame_counter / total_time_for_inference

    with open(out_path+'stats.txt', 'w') as f:
        f.write('Inference time: ' + str(total_time_for_inference) + '\n')
        f.write('FPS: ' + str(fps) + '\n')
        f.write('Model load time: ' + str(total_load_time) + '\n')

    loger.error("The video stream is over...")
    cv2.destroyAllWindows()
    feeder_in.close()
    
     
    

if __name__ == '__main__':
    main() 
