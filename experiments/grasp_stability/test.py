# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time

import deepdish as dd
import numpy as np
import pybullet as pb
import pybullet_data
import tacto  # import TACTO
from robot import Robot
import pybulletX as px

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs.log", level=logging.DEBUG)


class Camera:
    def __init__(self, cameraResolution=[320, 240]):
        self.cameraResolution = cameraResolution

        camTargetPos = [0.5, 0, 0.05]
        camDistance = 0.4
        upAxisIndex = 2

        yaw = 90
        pitch = -30.0
        roll = 0

        fov = 60
        nearPlane = 0.01
        farPlane = 100

        self.viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = cameraResolution[0] / cameraResolution[1]

        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )

    def get_image(self):
        img_arr = pb.getCameraImage(
            self.cameraResolution[0],
            self.cameraResolution[1],
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth data
        return rgb, dep


def get_forces(bodyA=None, bodyB=None, linkIndexA=None, linkIndexB=None):
    """
    get contact forces

    :return: normal force, lateral force
    """
    kwargs = {
        "bodyA": bodyA,
        "bodyB": bodyB,
        "linkIndexA": linkIndexA,
        "linkIndexB": linkIndexB,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    pts = pb.getContactPoints(**kwargs)

    totalNormalForce = 0
    totalLateralFrictionForce = [0, 0, 0]

    for pt in pts:
        totalNormalForce += pt[9]

        totalLateralFrictionForce[0] += pt[11][0] * pt[10] + pt[13][0] * pt[12]
        totalLateralFrictionForce[1] += pt[11][1] * pt[10] + pt[13][1] * pt[12]
        totalLateralFrictionForce[2] += pt[11][2] * pt[10] + pt[13][2] * pt[12]

    return totalNormalForce, totalLateralFrictionForce

def load_object(index) :
    parentDir = "/home/yan/Pybullet Objects/Pybullet Objects"
    obj_list = [
        {
            "urdfObj" : parentDir + "/Image Data/006_mustard_bottle/google_16k/bottle.urdf",
            "scaling" : 4,
            "pos" : [0.5, 0, 0.10],
            "name" : "bottle"
        },
        {
            "urdfObj" : parentDir + "/Image Data/002_master_chef_can/google_16k/can.urdf",
            "scaling" : 4,
            "pos" : [0.5, 0, 0.10],
            "name" : "can"
        },
        {
            "urdfObj" : parentDir + "/Image Data/013_apple/google_16k/apple.urdf",
            "scaling" : 6,
            "pos" : [0.5, 0, 0.10],
            "name" : "apple"
        },
        {
            "urdfObj" : parentDir + "/Image Data/011_banana/google_16k/banana.urdf",
            "scaling" : 7,
            "pos" : [0.5, 0, 0.10],
            "name" : "banana" 
        },
        {
            "urdfObj" : parentDir + "/Image Data/055_baseball/google_16k/baseball.urdf",
            "scaling" : 6,
            "pos" : [0.5, 0, 0.10],
            "name" : "baseball" 
        },
        {
            "urdfObj" : "setup/objects/cube_small.urdf",
            "scaling" : 0.6,
            "pos" : [0.5, 0, 0.205],
            "name" : "cube"
        }
    ]
    selectedObj = obj_list[index]
    urdfObj = selectedObj["urdfObj"]
    globalScaling = selectedObj["scaling"]
    pos = selectedObj["pos"].copy()
    name = selectedObj["name"]
    return urdfObj, globalScaling, pos, name

class Log:
    def __init__(self, dirName, id=0):
        self.dirName = dirName
        self.id = id
        self.dataList = []
        self.batch_size = 100
        os.makedirs(dirName, exist_ok=True)

    def save(
        self,
        tactileColorL,
        tactileColorR,
        tactileDepthL,
        tactileDepthR,
        visionColor,
        visionDepth,
        gripForce,
        normalForce,
        label,
    ):
        data = {
            "tactileColorL": tactileColorL,
            "tactileColorR": tactileColorR,
            "tactileDepthL": tactileDepthL,
            "tactileDepthR": tactileDepthR,
            "visionColor": visionColor,
            "visionDepth": visionDepth,
            "gripForce": gripForce,
            "normalForce": normalForce,
            "label": label,
        }

        self.dataList.append(data.copy())

        if len(self.dataList) >= self.batch_size:
            id_str = "{:07d}".format(self.id)
            # os.makedirs(outputDir, exist_ok=True)
            outputDir = os.path.join(self.dirName, id_str)
            os.makedirs(outputDir, exist_ok=True)

            # print(newData["tactileColorL"][0].shape)
            newData = {k: [] for k in data.keys()}
            for d in self.dataList:
                for k in data.keys():
                    newData[k].append(d[k])

            for k in data.keys():
                fn_k = "{}_{}.h5".format(id_str, k)
                outputFn = os.path.join(outputDir, fn_k)
                dd.io.save(outputFn, newData[k])

            self.dataList = []
            self.id += 1


px.init()
# Initialize World
logging.info("Initializing world")
physicsClient = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

# Initialize digits
digits = tacto.Sensor(width=240, height=320, visualize_gui=False)


pb.resetDebugVisualizerCamera(
    cameraDistance=0.6,
    cameraYaw=15,
    cameraPitch=-20,
    # cameraTargetPosition=[-1.20, 0.69, -0.77],
    cameraTargetPosition=[0.5, 0, 0.08],
)

planeId = pb.loadURDF("plane.urdf")  # Create plane

robotURDF = "setup/robots/sawyer_wsg50.urdf"
# robotURDF = "robots/wsg50.urdf"
robotID = pb.loadURDF(robotURDF, useFixedBase=True)
rob = Robot(robotID)


cam = Camera()
color, depth = cam.get_image()


rob.go(rob.pos, wait=True)

sensorLinks = rob.get_id_by_name(
    ["joint_finger_tip_left", "joint_finger_tip_right"]
)  # [21, 24]
digits.add_camera(robotID, sensorLinks)

nbJoint = pb.getNumJoints(robotID)

# Add object to pybullet and tacto simulator
# urdfObj = "setup/objects/cube_small.urdf"
urdfObj, globalScaling, pos, target_file_name = load_object(0)
objStartPos = [0.50, 0, 0.05]
objStartOrientation = pb.getQuaternionFromEuler([0, 0, np.pi / 2])

objID = digits.loadURDF(
    urdfObj, objStartPos, objStartOrientation, globalScaling=globalScaling
)

objStartPos2 = [0.50, 0.005, 0.05]
urdfObj2, globalScaling2, pos2, target_file_name2 = load_object(2)
obj2Id = digits.loadURDF(urdfObj2, objStartPos2, objStartOrientation, globalScaling=globalScaling)

sensorID = rob.get_id_by_name(["joint_finger_tip_right", "joint_finger_tip_left"])


def get_object_pose():
    res = pb.getBasePositionAndOrientation(objID)

    world_positions = res[0]
    world_orientations = res[1]

    if (world_positions[0] ** 2 + world_positions[1] ** 2) > 0.8 ** 2:
        pb.resetBasePositionAndOrientation(objID, objStartPos, objStartOrientation)
        return objStartPos, objStartOrientation

    world_positions = np.array(world_positions)
    world_orientations = np.array(world_orientations)

    return (world_positions, world_orientations)


time_render = []
time_vis = []

dz = 0.003
interval = 10
# posList = [
#     [0.50, 0, 0.205],
#     [0.50, 0, 0.213],
#     [0.50, 0.03, 0.205],
#     [0.50, 0.03, 0.213],
# ]
# posID = 0
# pos = posList[posID].copy()

t = 0
gripForce = 20

color, depth = digits.render()
digits.updateGUI(color, depth)

normalForceList0 = []
normalForceList1 = []

print("\n")
num_successes = 0
num_failures = 0
log = Log("data/grasp/" + target_file_name)
while True:
    # pick_and_place()
    t += 1
    result = pb.getContactPoints(objID, linkIndexA=-1) 

    for item in result :
        print("******",item[5])   
    print("******************************************") 
    # break
    pb.stepSimulation()

    st = time.time()
    # color, depth = digits.render()

    time_render.append(time.time() - st)
    time_render = time_render[-100:]
    # print("render {:.4f}s".format(np.mean(time_render)), end=" ")
    st = time.time()

    # digits.updateGUI(color, depth)

    time_vis.append(time.time() - st)
    time_vis = time_vis[-100:]

    # print("visualize {:.4f}s".format(np.mean(time_vis)))

    digits.update()

pb.disconnect()  # Close PyBullet
