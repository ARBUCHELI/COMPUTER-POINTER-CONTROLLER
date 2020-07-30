from openvino.inference_engine import IECore, IENetwork
import cv2
import logging

class Model:
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.input_shape = None
        self.out_blob = None
        self.out_shape = None
        self.net_plugin = None
        self.device = device
        self.extensions = extensions
        self.logger = logging.getLogger('fd')
        
        self.model_name = model_name
        self.model_xml = self.model_name
        self.model_bin = self.model_name.split('.')[0]+'.bin'


        try:
            self.plugin = IECore()
            self.net = IENetwork(model = self.model_xml, weights = self.model_bin)
        except Exception as e:
            self.logger.error("Error" + str(self.model_name) + str(e))
            raise ValueError("It was not possible to initialize the network. Please check if you entered the correct model path")
    
    def load_model(self):

        supported_layers = self.plugin.query_network(network = self.net, device_name = self.device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len (unsupported_layers) != 0 and self.device == 'CPU':
            print ("Unsupported layers were found: {}".format(unsupported_layers))
            print ("Check whether extensions are available to add to IECore")
            if not self.extensions == None:
                print ("Adding a CPU extension to solve the issue")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.net, device_name = self.device)
                unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
                if len (unsupported_layers) != 0:
                    print ("The application still found unsupported layers after adding the extension")
                    exit(1)
                print ("The issue was solved")
            else:
                print("Please enter the path to CPU extension")
                exit(1)
        
        try:
            self.net_plugin = self.plugin.load_network(network = self.net, device_name = self.device, num_requests = 1)
        except Exception as e:
            self.logger.error("Error. It was not possible to load the network"+str(self.model_name)+str(e))

    def predict(self):
        pass

    def wait(self):
        status = self.network.requests[0].wait(-1)
        return status
    
    def preprocess_input(self, image):
        pass

    def preprocess_output(self):
        pass