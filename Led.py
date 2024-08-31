#$ pip install tuya-bulb-control --upgrade
# pip3 install tuya-iot-py-sdk

from tuya_iot import TuyaOpenAPI, TUYA_LOGGER, openapi
import logging
import time

from Led_Values import ENDPOINT, ACCESS_ID, ACCESS_KEY, USERNAME, PASSWORD,DEVICE_ID

red = {'commands': [{'code': 'work_mode', 'value': "music"}]}
off = {'commands': [{'code': 'switch_led', 'value': False}]}
commands = {'commands': [{'code': 'switch_led', 'value': True}]}

def trigger_alert():
# Initialization of tuya openapi
    openapi = TuyaOpenAPI(ENDPOINT, ACCESS_ID, ACCESS_KEY)
    openapi.connect(USERNAME, PASSWORD,'972',"tuyasmart")
    TUYA_LOGGER.setLevel(logging.DEBUG)


    i=0
    while(i<10):
     i=i+1
     #commands =  {'commands': [{'code': 'switch_led', 'value': True}]}
     openapi.post('/v1.0/devices/{}/commands'.format(DEVICE_ID), commands)
     time.sleep(2)
     openapi.post('/v1.0/devices/{}/commands'.format(DEVICE_ID), off)
     time.sleep(2)