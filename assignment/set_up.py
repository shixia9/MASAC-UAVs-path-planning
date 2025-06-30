import pygame
from assignment import  constants as C
from assignment import tools
pygame.init()
pygame.mixer.init()
SCREEN=pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

GRAPHICS=tools.load_graphics('D:/drlTest/UAV-path-planning-main\Multi-UAVs path planning\path planning/assignment\source\image')

SOUND=tools.load_sound('D:/drlTest/UAV-path-planning-main\Multi-UAVs path planning\path planning/assignment\source\music')