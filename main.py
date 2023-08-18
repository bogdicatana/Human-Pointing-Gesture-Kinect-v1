import thread
import itertools
import ctypes
import _ctypes
import math
import numpy as np
import sys

import pykinect
from pykinect import nui
from pykinect.nui import JointId

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

# object locations
# (0, 0) is top left with the intervals being (0->640, 0->480)
# THESE ARE PROOF OF CONCEPT
SPOT_1_POSITION = (50, 250) # Replace with desired position for obj 1
SPOT_1_ANGLE = 40  # Replace with the desired angle for spot 1
SPOT_1_ANGLE_UP = 125
SPOT_2_POSITION = (600, 250) # Replace with desired position for obj 2
SPOT_2_ANGLE = 30  # Replace with the desired angle for spot 2
SPOT_3_POSITION = (50, 350)
SPOT_3_ANGLE_DOWN = 110

poscheck = False # this checks whether the experiment is set up properly
global arr2d

# Constants for joint indices
ELBOW_LEFT = JointId.ElbowLeft
SHOULDER_LEFT = JointId.ShoulderLeft
WRIST_LEFT = JointId.WristLeft
ELBOW_RIGHT = JointId.ElbowRight
SHOULDER_RIGHT = JointId.ShoulderRight
WRIST_RIGHT = JointId.WristRight
HIP_RIGHT = JointId.hip_right
HIP_LEFT = JointId.hip_left
SHOULDER_CENTER = JointId.ShoulderCenter

# Constants for screen dimensions
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Pygame color constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Global variables to store the elbow angle
elbow_angle_left = 0
elbow_angle_right = 0
armpit_angle_left = 0
armpit_angle_right = 0
point_angle_left = 0
point_angle_right = 0

# Kinect runtime object
kinect = None

KINECTEVENT = pygame.USEREVENT
DEPTH_WINSIZE = 640,480
VIDEO_WINSIZE = 640,480

tmp_s = pygame.Surface(DEPTH_WINSIZE, 0, 16)
pygame.init()

SKELETON_COLORS = [THECOLORS["red"], 
                   THECOLORS["blue"], 
                   THECOLORS["green"], 
                   THECOLORS["orange"], 
                   THECOLORS["purple"], 
                   THECOLORS["yellow"], 
                   THECOLORS["violet"]]

LEFT_ARM = (JointId.ShoulderCenter, 
            JointId.ShoulderLeft, 
            JointId.ElbowLeft, 
            JointId.WristLeft, 
            JointId.HandLeft)
RIGHT_ARM = (JointId.ShoulderCenter, 
             JointId.ShoulderRight, 
             JointId.ElbowRight, 
             JointId.WristRight, 
             JointId.HandRight)
LEFT_LEG = (JointId.HipCenter, 
            JointId.HipLeft, 
            JointId.KneeLeft, 
            JointId.AnkleLeft, 
            JointId.FootLeft)
RIGHT_LEG = (JointId.HipCenter, 
             JointId.HipRight, 
             JointId.KneeRight, 
             JointId.AnkleRight, 
             JointId.FootRight)
SPINE = (JointId.HipCenter, 
         JointId.Spine, 
         JointId.ShoulderCenter, 
         JointId.Head)

skeleton_to_depth_image = nui.SkeletonEngine.skeleton_to_depth_image

def calculate_angle(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0:
        return 0
    angle = np.degrees(np.arccos(dot_product / magnitude))
    if math.isnan(angle):
        return None
    return int(angle)

def draw_checkmark(surface, position, radius):
    pygame.draw.circle(surface, WHITE, position, radius, 2)
    pygame.draw.line(surface, THECOLORS["blue"], (position[0] - radius, position[1]), 
                     (position[0] - radius // 2, position[1] + radius // 2), 2)
    pygame.draw.line(surface, THECOLORS["blue"], (position[0] - radius // 2, 
                                      position[1] + radius // 2), (position[0] + radius, position[1] - radius), 2)

# PyKinect Skeleton Tracking Handler
def skeleton_frame_ready(frame):
    # Get the tracked skeleton
    skeleton = frame.SkeletonData[0]
    for i in frame.SkeletonData:
        if i.eTrackingState == nui.SkeletonTrackingState.TRACKED:
            skeleton = i
    # Get joint positions
    elbow_left = skeleton.SkeletonPositions[ELBOW_LEFT]
    shoulder_left = skeleton.SkeletonPositions[SHOULDER_LEFT]
    wrist_left = skeleton.SkeletonPositions[WRIST_LEFT]

    elbow_right = skeleton.SkeletonPositions[ELBOW_RIGHT]
    shoulder_right = skeleton.SkeletonPositions[SHOULDER_RIGHT]
    wrist_right = skeleton.SkeletonPositions[WRIST_RIGHT]

    # Convert joint positions to NumPy arrays
    elbow_left = np.array([elbow_left.x, elbow_left.y, elbow_left.z])
    shoulder_left = np.array([shoulder_left.x, shoulder_left.y, shoulder_left.z])
    wrist_left = np.array([wrist_left.x, wrist_left.y, wrist_left.z])

    elbow_right = np.array([elbow_right.x, elbow_right.y, elbow_right.z])
    shoulder_right = np.array([shoulder_right.x, shoulder_right.y, shoulder_right.z])
    wrist_right = np.array([wrist_right.x, wrist_right.y, wrist_right.z])

    hip_r = skeleton.SkeletonPositions[HIP_RIGHT]
    hip_l = skeleton.SkeletonPositions[HIP_LEFT]
    hip_r = np.array([hip_r.x, hip_r.y, hip_r.z])
    hip_l = np.array([hip_l.x, hip_l.y, hip_l.z])

    should_cent = skeleton.SkeletonPositions[SHOULDER_CENTER]
    should_cent = np.array([should_cent.x, should_cent.y, should_cent.z])

    # Calculate angle
    global elbow_angle_left, elbow_angle_right, armpit_angle_left, armpit_angle_right, point_angle_left, point_angle_right
    elbow_angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
    elbow_angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
    armpit_angle_right = calculate_angle(hip_r, elbow_right, shoulder_right)
    armpit_angle_left = calculate_angle(hip_l, elbow_left, shoulder_left)
    point_angle_left = calculate_angle(elbow_left, shoulder_left, should_cent)
    point_angle_right = calculate_angle(elbow_right, shoulder_right, should_cent)
    global poscheck
    if(elbow_angle_left < 20 and armpit_angle_left > 100 and poscheck):
        print("Pointing left at {} and down at {}".format(point_angle_left, armpit_angle_left))
        if abs(point_angle_left - SPOT_1_ANGLE) < 10:  # Adjust the angle tolerance as needed
            if abs(armpit_angle_left - SPOT_3_ANGLE_DOWN < 10):
                #print("pointing at 3")
                draw_checkmark(screen, SPOT_3_POSITION, 15)
            elif abs(armpit_angle_left - SPOT_1_ANGLE_UP < 10):
                #print("pointing at 1")
                draw_checkmark(screen, SPOT_1_POSITION, 15)
    if(elbow_angle_right < 20 and armpit_angle_right > 100):
        print("Pointing right at {} and down at {}".format(point_angle_right, armpit_angle_right))
        if abs(point_angle_right - SPOT_2_ANGLE) < 10:  # Adjust the angle tolerance as needed
            #print("pointing at 2")
            draw_checkmark(screen, SPOT_2_POSITION, 15)

def draw_skeleton_data(pSkelton, index, positions, width = 4):
    start = pSkelton.SkeletonPositions[positions[0]]
       
    for position in itertools.islice(positions, 1, None):
        next = pSkelton.SkeletonPositions[position.value]
        
        curstart = skeleton_to_depth_image(start, dispInfo.current_w, dispInfo.current_h) 
        curend = skeleton_to_depth_image(next, dispInfo.current_w, dispInfo.current_h)

        pygame.draw.line(screen, SKELETON_COLORS[index], curstart, curend, width)
        
        start = next

if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
   Py_ssize_t = ctypes.c_int
elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
   Py_ssize_t = ctypes.c_int64
else:
   raise TypeError("Cannot determine type of Py_ssize_t")

_PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
_PyObject_AsWriteBuffer.restype = ctypes.c_int
_PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                                  ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(Py_ssize_t)]

def surface_to_array(surface):
   buffer_interface = surface.get_buffer()
   address = ctypes.c_void_p()
   size = Py_ssize_t()
   _PyObject_AsWriteBuffer(buffer_interface,
                          ctypes.byref(address), ctypes.byref(size))
   bytes = (ctypes.c_byte * size.value).from_address(address.value)
   bytes.object = buffer_interface
   return bytes

def draw_skeletons(skeletons):
    global arr2d
    # check that the objects are in the right places and then display them
    # WILL ONLY DISPLAY THEM IF THERE IS A PERSON IN THE FRAME (at about 2 meters distance from camera)
    global poscheck
    count = 0 # counts the positions that are correctly set up
    if(arr2d[320,240]<=2100 and arr2d[320,240]>=1950):
        count+=1
        if(arr2d[SPOT_1_POSITION] <= 1520 and arr2d[SPOT_1_POSITION] >= 1420):
            count+=1 
            pygame.draw.circle(screen, THECOLORS["red"], SPOT_1_POSITION, 10, 0)
        if(arr2d[SPOT_2_POSITION] <= 1520 and arr2d[SPOT_2_POSITION] >= 1420):
            count+=1
            pygame.draw.circle(screen, THECOLORS["red"], SPOT_2_POSITION, 10, 0)
        if(arr2d[SPOT_3_POSITION] <= 1520 and arr2d[SPOT_3_POSITION] >= 1420):
            count+=1
            pygame.draw.circle(screen, THECOLORS["red"], SPOT_3_POSITION, 10, 0)
    if(count == 4): # change 4 to the number of positions you are checking (3 objects and 1 person in this case)
        poscheck = True
    for index, data in enumerate(skeletons):
        # draw the Head
        HeadPos = skeleton_to_depth_image(data.SkeletonPositions[JointId.Head], dispInfo.current_w, dispInfo.current_h) 
        draw_skeleton_data(data, index, SPINE, 10)
        pygame.draw.circle(screen, SKELETON_COLORS[index], (int(HeadPos[0]), int(HeadPos[1])), 20, 0)
    
        # drawing the limbs
        draw_skeleton_data(data, index, LEFT_ARM)
        draw_skeleton_data(data, index, RIGHT_ARM)
        draw_skeleton_data(data, index, LEFT_LEG)
        draw_skeleton_data(data, index, RIGHT_LEG)


def depth_frame_ready(frame):
    if video_display:
        return
    global arr2d
    with screen_lock:
        frame.image.copy_bits(tmp_s._pixels_address)
        arr2d = (pygame.surfarray.pixels2d(tmp_s) >> 3) & 4095
        # arr2d[x,y] is depth data in mm at (x,y)
        # Convert the 16-bit depth data to 8-bit grayscale
        arrr2d = arr2d.astype('uint8')
        # Create a Pygame surface from the 8-bit grayscale data
        depth_surface = pygame.surfarray.make_surface(arrr2d)

        # Scale the surface to match the screen size and display it
        screen.blit(pygame.transform.scale(depth_surface, DEPTH_WINSIZE), (0, 0))
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        pygame.display.update()    

def video_frame_ready(frame):
    if not video_display:
        return

    with screen_lock:
        address = surface_to_array(screen)
        frame.image.copy_bits(address)
        del address
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        pygame.display.update()

if __name__ == '__main__':
    full_screen = False
    draw_skeleton = True
    video_display = False

    screen_lock = thread.allocate()
    if video_display:
        screen = pygame.display.set_mode(VIDEO_WINSIZE,0,32)
    else:
        screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
    pygame.display.set_caption('Python Kinect Demo')
    skeletons = None
    screen.fill(THECOLORS["black"])

    kinect = nui.Runtime()
    kinect.skeleton_engine.enabled = True
    def post_frame(frame):
        try:
            pygame.event.post(pygame.event.Event(KINECTEVENT, skeletons = frame.SkeletonData))
        except:
            # event queue full
            pass

    kinect.skeleton_frame_ready += post_frame
    
    kinect.depth_frame_ready += depth_frame_ready    
    kinect.video_frame_ready += video_frame_ready    
    
    kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)

    print('Controls: ')
    print('     d - Switch to depth view')
    print('     v - Switch to video view')
    print('     s - Toggle displaing of the skeleton')
    print('     u - Increase elevation angle')
    print('     j - Decrease elevation angle')

    # main game loop
    done = False
    kinect.skeleton_frame_ready += skeleton_frame_ready

    while not done:
        e = pygame.event.wait()
        dispInfo = pygame.display.Info()
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == KINECTEVENT:
            skeletons = e.skeletons
            if draw_skeleton:
                draw_skeletons(skeletons)
                pygame.display.update()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                done = True
                break
            elif e.key == K_d:
                with screen_lock:
                    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
                    video_display = False
            elif e.key == K_v:
                with screen_lock:
                    screen = pygame.display.set_mode(VIDEO_WINSIZE,0,32)    
                    video_display = True
            elif e.key == K_s:
                draw_skeleton = not draw_skeleton
            elif e.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                kinect.camera.elevation_angle = 2