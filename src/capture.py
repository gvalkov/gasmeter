#!/usr/bin/env python
# -*- coding: utf-8; -*-


from time import sleep
from abc import ABCMeta, abstractmethod

import cv2


#-----------------------------------------------------------------------------
class CaptureInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def capture(self):
        pass

    @abstractmethod
    def setup(self):
        pass


class Webcam(CaptureInterface):
    webcam = None  # cv2.VideoCapture()

    def __init__(self, device, led_gpio_pin, wh=(640, 480)):
        self.device = device
        self.led_gpio_pin = led_gpio_pin
        self.wh = wh
        self.webcam = self.setup()

    def setup(self):
        import RPi.GPIO
        self.GPIO = GPIO = RPi.GPIO

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.led_gpio_pin, GPIO.OUT)

        webcam = cv2.VideoCapture(self.device)
        webcam.set(3, self.wh[0])
        webcam.set(4, self.wh[1])
        return webcam

    def capture(self):
        try:
            self.GPIO.output(self.led_gpio_pin, True)
            sleep(0.3)
            ret, img = self.webcam.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        finally:
            sleep(1.0)
            self.GPIO.output(self.led_gpio_pin, False)

    def __call__(self):
        return self.capture()
