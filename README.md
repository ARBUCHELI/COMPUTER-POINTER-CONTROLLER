# Computer Pointer Controller

This project makes use of a gaze detection model, to control the computer mouse pointer.  This project demonstrates OpenVino's ability to run multiple models on the 
same machine and coordinate the flow of data between those models.

## How the project works

The gaze estimation model requires three inputs:

	* The head pose.
	* The left eye image.
	* The right eye image.

To get these inputs, it's necessary to use three other OpenVino models:

	* Face Detection
	* Head Pose Estimation
	* Facial Landmarks Detection.

Please take a look to the outputs folder located inside the final_project folder to see pictures and a short video that show how the project works.

## Project Set Up and Installation

The final_project folder contains 8 necessary python files to run the project.  The code to run the application is contained in the main.py file and this file
integrates the interaction of the other python files.  

The Directory Structure is as Following: 

Explanation: 

- The Benchmarks folder contains documents with the outputs of the different tests of the app with different model precisions (The original name for each one of this
outputs is results and the files were saved with a different name).

- The Commands.txt file is an explanatory document with the different commands used to deploy the app.

- The demo.mp4 file is the video used to test the app.

- The main.py file makes use and imports classes and methods from the other 7 python files.

- The model.py file contains classes and methods used to load the 4 models.

- The outputs folder contains pictures and a short video that demonstrates how the app works.

- The requirements.txt file contains all the required dependencies.

- The __pycache__ folder contains files created by the Python interpreter used to skip in subsequent import, the translation from source code to bytecode.
- The env folder contains files generated by Python.

	<DIR> final_project
	      |
	       __<DIR> Benchmarks
	      |            |___	CPU1.txt
	      |  	   |___	CPU10.txt
	      | 	   |___ CPU11.txt
	      | 	   |___ CPU12.txt
	      | 	   |___ CPU2.txt
	      |		   |___ CPU3.txt
	      |		   |___ CPU4.txt
	      |		   |___ CPU5.txt
	      |		   |___ CPU6.txt
	      |		   |___ CPU7.txt
              |		   |___ CPU8.txt
	      |		   |___ CPU9.txt	
	      |
              |
              |___ Commands.txt
	      |___ demo.mp4
	      |___ face_detection.py
	      |___ facial_landmarks_detection.py
	      |___ gaze_estimation.py
	      |___ head_pose_estimation.py
              |___ input_feeder.py
              |___ main.py
	      |___ model.py		
              |___ mouse_controller.py
              |
	      |___ <DIR>outputs
	      | 	|___ 20200728_064531_001.mp4
	      | 	|___ cameraoutput.png
	      |  	|___ videoutput.png
              |___ <DIR>env
	      |
	      |___ README.md
	      |___ requirements.txt
              |___ Untitled Document 1
	      |	
	      |___ <DIR>__pycache__
			     |___ face_detection.cpython-36.pyc
			     |___ facial_landmarks_detection.cpython-36.pyc
			     |___ gaze_estimation.cpython-36.pyc
			     |___ head_pose_estimation.cpython-36.pyc
			     |___ input_feeder.cpython-36.pyc
			     |___ mouse_controller.cpython-36.pyc


In order to set up and install the project its necessary to install the OpenVino Toolkit in the Operative System that the user prefers. The use of Linux Ubuntu is
highely recommended to avoid problems that can occur with the installation of additional software resources.

STEPS:
	1. Review the Software and Hardware required to run the OpenVino toolkit in your computer here:
	   https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/documentation.html

	2. Download OpenVino Toolkit (Ubuntu 18.04 Operative System Highly Recommended).
	
	3. Unzip and Istall the OpenVino Toolkit following the next commands in order to setup the development environment:
		tar -xvzf l_openvino_toolkit_p_2019.3.376.tgz
		cd l_openvino_toolkit_p_2019.3.376
		sudo ./install.sh
		cd l_openvino_toolkit_p_2019.3.376
		sudo -E ./install_openvino_dependencies.sh
		source /opt/intel/openvino/bin/setupvars.sh
		cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
		sudo ./install_prerequisites.sh
		cd /opt/intel/openvino/deployment_tools/demo	
                ./demo_squeezenet_download_convert_run.sh
                ./demo_security_barrier_camera.sh

	4. Source the OpenVino Development environment with the following command:
		source /opt/intel/openvino/bin/setupvars.sh

	5. Download the models from the openvino model zoo, using the model downloader:

		a. Go to the model downloader location in your pc with the following command:
			cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

		b. Download the models with the following commands. (The -o option will allow you to choose the location where you want to download the models.  A
		folder with the name <intel> containing will be created.  Pay attention to the name of the folder because the location of this folder containint the 
		.xml files of the models is very important to run the application).

		VERY IMPORTANT NOTE: Is highly recommended to download the models in the same folder that contains the application (final_project).

			sudo ./downloader.py --name face-detection-adas-binary-0001 -o /home/andres/P3/
			sudo ./downloader.py --name head-pose-estimation-adas-0001 -o /home/andres/P3/
			sudo ./downloader.py --name landmarks-regression-retail-0009 -o /home/andres/P3/
			sudo ./downloader.py --name gaze-estimation-adas-0002 -o /home/andres/P3/

	6. Depending on the setup of your Ubuntu operative system, you may have to install additional libraries. Install pyautogui and Tkinter with the following 
           commands:
			pip3 install pyautogui 
			sudo apt-get dist-upgrade
			apt-get install python-tk 

## Demo

Follow the next steps in order to run a demo of the project:

	1. Source the OpenVino Development Environment. (source /opt/intel/openvino/bin/setupvars.sh).

	2. Navigate with cd to the folder that contains the application (final_project).

	3. Set up a Virtual env (Ubuntu 18.04) Using the following commands:
		
		sudo apt-get update && sudo apt-get upgrade
		sudo apt install virtualenv
		mkdir ~/python-environments && cd ~/python-environments
		virtualenv --python=python3 env
		ls env/lib
		source env/bin/activate

	You will know that you are working on a virtual environment when you see something similar to this in your terminal (env)

		(env) andres@andres-HP-Notebook:~$ 

	VERY IMPORTANT NOTE:
 
	- Notice that the folder with the name <intel> in the case of the command used in the step 3 is located in the same directory that contains
	the project (final_project).  If you decided to download the models in a different directory, it will be necessary to change the paths to the different .xml
	files or to paste the folder with the name <intel> that contains the downloaded models in the directory that contains the project (final_project).

	- The video file is located in the final_project directory.  In the case that you want to test the app with a different video file, you need to specify the
	path to the video file that you want to test.

	4. Install opencv on ubuntu: sudo apt install python3-opencv
	
	5. Install pyautogui: pip3 install pyautogui
	
	5. Update your numpy installation: pip install -U numpy

	3. Run the application using the following command:

	python3 main.py -dt intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml -pe intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -ld intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -ge intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml  -i demo.mp4 -f fad hpe fld gad  -d CPU -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -o results
	
	VERY IMPORTANT NOTE: 
	- Don't forget to include the -l command line argument because some models have unsupported layers that need be be handled using a cpu extension.
	- CPU extension: /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so
	
## Documentation

The following command line arguments are necessary to run the application:

	-dt (Path to the face detection model .xml file).
	-pe (Path to the head pose estimation model .xml file).
	-ld (Path to the facial landmarks detection model .xml file).
	-ge (Path to the gaze estimation model .xml file).

The following command line argument has two options:

	a) -i <path to the video file> (If you want to test the application with a video file).
	b) cam (If you want to test the application with the cam of your computer).

The following command line argument is optional and it has four options (fad hpe fld gad). You can use each option separately or you can enter the four options 
separated by space.  The purpose of this options is to visualize the outputs of each model or all the outpus at the same time.

	a) -f fad (for face detection model)
	b) -f hpe (for head pose estimation model)
	c) -f fld (for facial landmarks detection model)
	d) -f gad (for gaze estimation model)
	e) -f fad hpe fld gad (For seeing all the outputs at the same time).

The following command line argument allows the user to specify the type of device:

	-d CPU (This is an example of how to use the command line argument to try the application in a CPU.  Change to a different option if you want to try the
	application in other devices).

	Options: GPU, FPGA, MYRIAD.

The following command line is used to specify the name of a .txt document with the results of performance statistics of the application. (This is an optional command):

	-o results (Example of usage for a .txt file with the name <results>.

## Benchmarks

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP16
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: FP16
Inference time: 25.6
FPS: 2.3046875
Model load time: 0.6028685569763184
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP16
LANDMARK DETECTION MODEL: FP32
GAZE ESTIMATION MODEL: FP16
Inference time: 25.8
FPS: 2.2868217054263567
Model load time: 0.6019582748413086
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP16
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: FP32
Inference time: 25.9
FPS: 2.2779922779922783
Model load time: 0.58302903175354
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP16
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: INT8
Inference time: 25.7
FPS: 2.295719844357977
Model load time: 0.6072094440460205
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP16
LANDMARK DETECTION MODEL: FP32
GAZE ESTIMATION MODEL: FP32
Inference time: 26.0
FPS: 2.269230769230769
Model load time: 0.5905206203460693
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP16
LANDMARK DETECTION MODEL: FP32
GAZE ESTIMATION MODEL: INT8
Inference time: 25.3
FPS: 2.3320158102766797
Model load time: 0.8243587017059326
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32           
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: FP16
Inference time: 25.2               
FPS: 2.3412698412698414            
Model load time: 0.5792114734649658
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: FP32
Inference time: 25.8
FPS: 2.2868217054263567
Model load time: 0.5617036819458008 
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: INT8
Inference time: 25.5
FPS: 2.3137254901960786
Model load time: 0.6087915897369385
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32
LANDMARK DETECTION MODEL: FP32
GAZE ESTIMATION MODEL: FP16
Inference time: 25.7
FPS: 2.295719844357977
Model load time: 0.5800726413726807
___________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32
LANDMARK DETECTION MODEL: FP32
GAZE ESTIMATION MODEL: FP32
Inference time: 25.8
FPS: 2.2868217054263567
Model load time: 0.5631494522094727
____________________________________

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32
LANDMARK DETECTION MODEL: FP32
GAZE ESTIMATION MODEL: INT8
Inference time: 25.4
FPS: 2.3228346456692917
Model load time: 0.5977311134338379
____________________________________


## Results

All the following results were tested in a Intel(R) Core(TM) i3-5005U CPU 2.00 GHz

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32           
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: FP16
Inference time: 25.2               --------> THIS IS THE COMBINATION OF MODEL PRECISIONS WITH THE BIGGEST INFERENCE TIME.
FPS: 2.3412698412698414            --------> THIS IS THE COMBINATION OF MODEL PRECISIONS THAT PROCESSES MORE FRAMES POR SECOND.
Model load time: 0.5792114734649658
___________________________________

The advantage of having FP16 at the end of the Pipeline, is that FP16 has the capability of improving speed and performance, and the data transfer is higher than 
FP32.  For that reason this combination is the faster one, but using two FP16 model precissions in the total pipeline increases the loading time because its necessary to make
a conversion from 32-bit floats.

See (https://software.intel.com/content/www/us/en/develop/articles/should-i-choose-fp16-or-fp32-for-my-deep-learning-model.html).

FACE DETECTION MODEL: INT 1
HEAD POSE ESTIMATION: FP32
LANDMARK DETECTION MODEL: FP16
GAZE ESTIMATION MODEL: FP32
Inference time: 25.8
FPS: 2.2868217054263567
Model load time: 0.5617036819458008 ------> THIS IS THE COMBINATION OF MODEL PRECISIONS THAT TAKES LESS TIME TO LOAD THE MODELS.
___________________________________

Using a Pipeline with two FP32 model precissions, is faster for loading but the performance is not as good as the previous combination of model precissions.





