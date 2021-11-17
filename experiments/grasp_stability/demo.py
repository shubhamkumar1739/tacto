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
from train import Model, GraspingDataset
from torchvision import transforms
import torch
import cv2

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
urdfObj, globalScaling, pos, target_file_name = load_object(5)
objStartPos = [0.50, 0, 0.05]
objStartOrientation = pb.getQuaternionFromEuler([0, 0, np.pi / 2])

objID = digits.loadURDF(
    urdfObj, objStartPos, objStartOrientation, globalScaling=globalScaling
)

objStartPos2 = [0.50, 0.05, 0.0]
urdfObj2, globalScaling2, pos2, target_file_name2 = load_object(2)
obj2Id = digits.loadURDF(urdfObj2, objStartPos2, objStartOrientation, globalScaling=globalScaling)

sensorID = rob.get_id_by_name(["joint_finger_tip_right", "joint_finger_tip_left"])


def get_object_pose(objID):
    res = pb.getBasePositionAndOrientation(objID)

    world_positions = res[0]
    world_orientations = res[1]

    if (world_positions[0] ** 2 + world_positions[1] ** 2) > 0.8 ** 2:
        pb.resetBasePositionAndOrientation(objID, objStartPos, objStartOrientation)
        return objStartPos, objStartOrientation

    world_positions = np.array(world_positions)
    world_orientations = np.array(world_orientations)

    return (world_positions, world_orientations)

transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
transformDepth = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1,), std=(0.2,)),
                # AddGaussianNoise(0.0, 0.01),
            ]
        )
# testDataset = GraspingDataset(
#             testFileNames,
#             fields=["tactileColorL", "tactileColorR", "visionColor"],
#             transform=transform,
#             transformDepth=transformDepth,
#         )
# testLoader = torch.utils.data.DataLoader(
#     testDataset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True
# )

def get_prediction(model, x) :
    for k in x :
        x[k] = x[k][:, :, :3]
        x[k] = transform(x[k])
        x[k] = torch.reshape(x[k], [1, 3, 224, 224])       # print(k, x[k].shape, x[k].type)
    prediction = None
    with torch.no_grad() :
        prediction = model(x)
        prediction = prediction.argmax(axis=-1)
        # print(prediction)
    return prediction

def reset(environ) :
    object_list = environ["object_id_list"]
    rob = 
    for objId in object_list :
        objStartPos, objStartOrientation = get_obj_configuration(objId)
        pb.resetBasePositionAndOrientation(objID, objStartPos, objStartOrientation)
        for i in range(100) :
            pb.stepSimulation()

    robStartPos = environ["robStartPos"]
    robStartOrientation = environ["robStartOrientation"]
    rob.go(robStartPos, ori=robStartOrientation, width=0.11)
    for i in range(100) :
        pb.stepSimulation()

def get_feasibility(environ, models) :
    feasibility_socres = []
    num_attempts = 100
    for i in range(num_attempts) :
        score = attempt_grasp(environ, models)
        if score is not None :
            feasibility_socres.append(score)
    reset(environ)
    return feasibility_socres


def get_approach_pose_and_force(target_obj) :
    pass

def attempt_grasp(environ, models) :
    t = 0
    reset(environ)
    target_obj = environ["target_obj_id"]
    object_list = environ["object_id_list"]
    rob = environ["rob"]
    model = models[target_obj]

    pos, gripForce = get_approach_pose_and_force(target_obj)

    prediction = None
    graspPos = None
    graspOri = None

    while t < 2000 :
        t += 1
        if t <= 50 :
            rob.go(pos, width=0.11)
        elif t < 200 :
            rob.go(pos, width=0.03, gripForce=gripForce)
        elif t == 200 :
            tactileColor, tactileDepth = digits.render()
            tactileColorL, tactileColorR = tactileColor[0], tactileColor[1]

            visionColor, visionDepth = cam.get_image()

            digits.updateGUI(tactileColor, tactileDepth)

            normalForce0, lateralForce0 = get_forces(robotID, objID, sensorID[0], -1)
            normalForce1, lateralForce1 = get_forces(robotID, objID, sensorID[1], -1)
            normalForce = [normalForce0, normalForce1]

            x = {"tactileColorL" : tactileColorL, 
            "tactileColorR" : tactileColorR, 
            "visionColor" : visionColor}

            # cv2.imshow("img", visionColor)
            # cv2.waitKe
            prediction = get_prediction(model, x)
            graspPos, graspOri = get_object_pose()

        contact_data = pb.getContactPoints(robotID)
        collision = False
        for item in contact_data :
            if item[2] in object_list and item[2] != target_obj:
                collision = True
        if collision :
            return None

        pb.stepSimulation()
        digits.update()
    return prediction, graspPos, graspOri

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
objectList = [obj2Id]

num_attempts = 0
num_collisions = 0

model_path = "models/grasp/" + target_file_name + "/field['tactileColorL', 'tactileColorR', 'visionColor']_N120_i4.pth"
# model = torch.load(model_path)
model = Model(["tactileColorL", "tactileColorR", "visionColor"])
model.load(model_path)
model.eval()
print("Model created")

print("\n")
log = Log("data/grasp/" + target_file_name)
while num_attempts < 10000 and num_collisions < 10000:
    # pick_and_place()
    t += 1
    if t <= 50 :
        rob.go(pos, width=0.11)
    elif t < 200 :
        rob.go(pos, width=0.03, gripForce=gripForce)
    elif t == 200 :
        tactileColor, tactileDepth = digits.render()
        tactileColorL, tactileColorR = tactileColor[0], tactileColor[1]
        tactileDepthL, tactileDepthR = tactileDepth[0], tactileDepth[1]

        visionColor, visionDepth = cam.get_image()

        digits.updateGUI(tactileColor, tactileDepth)

        normalForce0, lateralForce0 = get_forces(robotID, objID, sensorID[0], -1)
        normalForce1, lateralForce1 = get_forces(robotID, objID, sensorID[1], -1)
        normalForce = [normalForce0, normalForce1]

        x = {"tactileColorL" : tactileColorL, 
        "tactileColorR" : tactileColorR, 
        "visionColor" : visionColor}

        # cv2.imshow("img", visionColor)
        # cv2.waitKe
        prediction = get_prediction(model, x)
        # print(prediction)

        objPos0, objOri0 = get_object_pose()
    elif t <= 260 :
        pos[-1] += dz
        rob.go(pos)
    elif t > 340 :
        objPos, objOri = get_object_pose()

        if objPos[2] - objPos0[2] < 60 * dz * 0.8:
            # Fail
            label = 0
        else:
            # Success
            label = 1

        if prediction == label :
            print("Correct prediction", prediction)
        else :
            print("Prediction", prediction, "label: ", label)

        num_attempts += 1
    
        if label == 1 :
            # some code
            pass

        t = 0

        # rob.go(pos, width=0.11)
        # for i in range(100):
        #     pb.stepSimulation()
        objRestartPos, objRestartOrientation, pos, ori, gripForce = restart()

pb.disconnect()  # Close PyBullet
