 # Import required libraries
import cv2 as cv
import torch
import numpy as np
import urllib.request
from gpiozero import LED
import RPi.GPIO as GPIO
import time
import subprocess

class setUp:
    def __init__(self):
       
        self.model = None
        self.imgResponse = None

       
        # Set a default state of the two modes
        self.state = False
        # Set a default image number
        self.image_count = 0
        # Set a default trigger state
        self.state_automat = 0

        # Define some constants for pin numbers
        self.AUTOMAT_PIN = 33
        self.READY_PIN = 17
        self.RESULT_READY_PIN = 35
        self.CORRECT_PIN = 18
        self.DEFECTIVE1_PIN = 27
        self.SIGNAL_PIN = 22
        self.SIGNAL_READY_PIN = 24


        
       

        # Set up the GPIO pins
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.RESULT_READY_PIN, GPIO.OUT)       
        GPIO.setup(self.SIGNAL_PIN, GPIO.OUT)
        GPIO.setup(self.SIGNAL_READY_PIN, GPIO.OUT)
        GPIO.setup(self.AUTOMAT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        # Set up the leds' pins
        self.led_ready = LED(self.READY_PIN)
        self.led_correct = LED(self.CORRECT_PIN)
        self.led_defective1 = LED(self.DEFECTIVE1_PIN)
        

        # Turn off the LEDs
        self.led_ready.off()
        self.led_correct.off()
        self.led_defective1.off()
       

        # Load the custom-trained computer vision model
        self.model = torch.hub.load('.', 'custom', path='bestV2.pt', source='local')
        # Define the confidence for the model
        self.model.conf = 0.85

        # Set up an interrupt for the trigger
        GPIO.add_event_detect(self.AUTOMAT_PIN, GPIO.RISING, callback=self.change_image_count, bouncetime=500)
       
    
        
    # This function change the image_count and the state_automat to 1
    def change_image_count(self, AUTOMAT_PIN):
        self.image_count = 1
        self.state_automat = 1

    
    # This function check if the esp32-cam is connect to the wifi or not
    def check_connection(self):
        connected = False
        while not connected:
            try:
                self.imgResponse = urllib.request.urlopen("http://192.168.43.117/capture?", timeout=5)
                self.control_led('100')
                connected = True
            except:
                self.control_led("000")
           
    # This function return the photo taken from the esp32-cam and its predictions
    def predict(self):
        imgNp = np.array(bytearray(self.imgResponse.read()), dtype=np.uint8)
        frame = cv.imdecode(imgNp, -1)
        return frame, self.model(frame)
    # This function  control the leds based on state_code
    def control_led(self, state_code):
         if state_code == "000":
            print("failed to connect the esp32")
            self.led_ready.off()
            self.led_correct.off()
            self.led_defective1.off()
            GPIO.output(self.SIGNAL_READY_PIN, 0)  
         elif state_code =="100":
            self.led_ready.on()
            GPIO.output(self.SIGNAL_READY_PIN, 1)  
            print("esp32 is connected")
         elif state_code == "010":
            print("Normal")
            self.led_correct.on()
            self.led_defective1.off()
            GPIO.output(self.SIGNAL_PIN, 1)  
         elif state_code == "001":
            print("defect")
            self.led_correct.off()
            self.led_defective1.on()
            GPIO.output(self.SIGNAL_PIN, 0)
        
            
                        
            
         else:
            print("Invalid state code")

    def show_result(self, result, width):
        bbox_height_part = None
        bbox_height_normal = None
        if result.pandas().xyxy[0].empty:
            self.control_led("001")
        else:
            for index, row in result.pandas().xyxy[0].iterrows():
                xmin, ymin, xmax, ymax = row[0], row[1], row[2], row[3]
                center_x = (xmin + xmax) / 2
                if center_x < width / 2:
                    self.control_led("001")
                if row[6] == "part":
                    x1, y1, x2, y2 = row[:4].astype(int)
                    bbox_height_part = y2 - y1
                elif row[6] == "Normal":
                    x1n, y1n, x2, y2n = row[:4].astype(int)
                    bbox_height_normal = y2n - y1n
                if bbox_height_part and bbox_height_normal:
                    ratio_word = bbox_height_part / bbox_height_normal
                    ratio_position = (y1 - y1n) * 100 // (y2n - y1n)
                    bbox_height_part = None
                    bbox_height_normal = None
                    if ratio_word > 0.51 or ratio_position > 40 :
                        self.control_led("001")
                    else:
                        self.control_led("010")   
                      

                if row[6] == "Inverted":
                        self.control_led("001")

    
        

    def optimal_mode(self):
        
        frame, result = self.predict()
        height, width, _ = frame.shape
        self.show_result(result, width)
        GPIO.output(self.RESULT_READY_PIN, 1)
        #state_ready = GPIO.input(self.RESULT_READY_PIN)
        #print("State of GPIO pin 35: {}".format(state_ready))
        time.sleep(0.25)
        GPIO.output(self.RESULT_READY_PIN, 0)
        #state_ready = GPIO.input(self.RESULT_READY_PIN)
        #print("State of GPIO pin 35: {}".format(state_ready))
        return 0 , 0