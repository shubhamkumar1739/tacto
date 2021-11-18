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

randOri = [0, 0, np.pi]
def load_object(objName) :
    parentDir = "/home/yan/Pybullet Objects/Pybullet Objects"
    obj_list = {
        "bottle" : {
            "urdfObj" : parentDir + "/Image Data/006_mustard_bottle/google_16k/bottle.urdf",
            "scaling" : 4,
            "pos" : [0.5, 0.1, 0],
            "ori" : randOri,
            "name" : "bottle"
        },
        "can" : {
            "urdfObj" : parentDir + "/Image Data/002_master_chef_can/google_16k/can.urdf",
            "scaling" : 4,
            "pos" : [0.55, 0.15, 0.0],
            "ori" : randOri,
            "name" : "can"
        },
        "apple" : {
            "urdfObj" : parentDir + "/Image Data/013_apple/google_16k/apple.urdf",
            "scaling" : 6,
            "pos" : [0.45, 0.13, 0.0],
            "ori" : randOri,
            "name" : "apple"
        },
        "banana" : {
            "urdfObj" : parentDir + "/Image Data/011_banana/google_16k/banana.urdf",
            "scaling" : 7,
            "pos" : [0.45, 0.05, 0],
            "ori" : randOri,
            "name" : "banana" 
        },
        "baseball" : {
            "urdfObj" : parentDir + "/Image Data/055_baseball/google_16k/baseball.urdf",
            "scaling" : 6,
            "pos" : [0.5, 0.1, 0],
            "ori" : randOri,
            "name" : "baseball" 
        },
        "cube" : {
            "urdfObj" : "setup/objects/cube_small.urdf",
            "scaling" : 0.6,
            "pos" : [0.5, 0, 0],
            "ori" : randOri,
            "name" : "cube"
        }
    }
    selectedObj = obj_list[objName]
    urdfObj = selectedObj["urdfObj"]
    globalScaling = selectedObj["scaling"]
    pos = selectedObj["pos"].copy()
    ori = pb.getQuaternionFromEuler(selectedObj["ori"].copy())
    name = selectedObj["name"]
    return urdfObj, globalScaling, pos, ori, name

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

def create_objects(conf_dict) :
    for objName in conf_dict :
        objUrdf, globalScaling, objStartPos, objStartOrientation, target_file_name = load_object(objName)
        objId = pb.loadURDF(objUrdf, objStartPos, objStartOrientation, globalScaling=globalScaling)
        conf_dict[objName]["objId"] = objId
        conf_dict[objName]["objStartPos"] = objStartPos
        conf_dict[objName]["objStartOrientation"] = objStartOrientation
        conf_dict[objName]["globalScaling"] = globalScaling
        conf_dict[objName]["target_file_name"] = target_file_name

def get_obj_configuration(objId, conf_dict) :
    objStartPos = None
    objStartOrientation = None
    for objName in conf_dict :
        if conf_dict[objName]["objId"] == objId :
            objStartPos = conf_dict[objName]["objStartPos"]
            objStartOrientation = conf_dict[objName]["objStartOrientation"]
    return objStartPos, objStartOrientation

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

def get_prediction(model, x, transform) :
    for k in x :
        x[k] = x[k][:, :, :3]
        x[k] = transform(x[k])
        x[k] = torch.reshape(x[k], [1, 3, 224, 224]).to(torch.device("cuda:0"))       # print(k, x[k].shape, x[k].type)
    prediction = None
    print(x)
    with torch.no_grad() :
        prediction = model(x)
        prediction = prediction.argmax(axis=-1)
    return prediction

def reset(environ) :
    object_list = environ["object_id_list"]
    rob = environ["rob"]
    conf_dict = environ["conf_dict"]
    rob.reset_robot()
    robStartPos = environ["robStartPos"]
    rob.go([0.5, 0, 0.2], width=0.11)
    for i in range(100) :
        pb.stepSimulation()
    for objId in object_list :
        objStartPos, objStartOrientation = get_obj_configuration(objId, conf_dict)
        pb.resetBasePositionAndOrientation(objId, objStartPos, objStartOrientation)

    for i in range(100) :
        pb.stepSimulation()

def get_feasibility(environ, model) :
    feasibility_socres = []
    num_attempts = 10
    for i in range(num_attempts) :
        score = attempt_grasp(environ, model)
        if score is not None :
            feasibility_socres.append(score)
    reset(environ)
    return feasibility_socres


def get_approach_pose_and_force(environ, target_obj) :
    # minAABB, maxAABB = pb.getAABB(target_obj)
    # random_x = (maxAABB[0] + minAABB[0]) / 2
    # random_y = (minAABB[1] + maxAABB[1]) / 2
    # random_z = 0.1
    conf_dict = environ["conf_dict"]
    objRestartPos = conf_dict["cube"]["objStartPos"]
    pos = [
            objRestartPos[0] + np.random.uniform(-0.02, 0.02),
            objRestartPos[1] + np.random.uniform(-0.02, 0.02),
            objRestartPos[2] * (1 + np.random.random() * 0.5) + 0.14,
        ]

    gripForce = 5 + np.random.random() * 15
    return pos, gripForce

def attempt_grasp(environ, model) :
    t = 0
    reset(environ)
    target_obj = environ["target_obj_id"]
    object_list = environ["object_id_list"]
    transform = environ["transform"]
    rob = environ["rob"]
    digits = environ["digits"]
    sensorID = environ["sensorID"]
    robotID = environ["robotID"]
    cam = environ["cam"]

    pos, gripForce = get_approach_pose_and_force(environ, target_obj)

    prediction = None
    graspPos = None
    graspOri = None

    dz = 0.003

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

            x = {"tactileColorL" : tactileColorL, 
            "tactileColorR" : tactileColorR, 
            "visionColor" : visionColor}

            prediction = get_prediction(model, x, transform)
            graspPos, graspOri = get_object_pose(target_obj)

        elif t > 200 and t <= 260:
            # Lift
            pos[-1] += dz
            rob.go(pos)
        elif t > 340:
            # Save the data
            objPos, objOri = get_object_pose(target_obj)

            if objPos[2] - graspPos[2] < 60 * dz * 0.8:
                # Fail
                label = 0
                # num_failures += 1
            else:
                # Success
                label = 1
                # num_successes += 1

            if label == prediction :
                print(True)
            else :
                print(False)
            break

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

def main() :
    px.init()
    # Initialize World
    physicsClient = pb.connect(pb.DIRECT)
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

    sensorID = rob.get_id_by_name(["joint_finger_tip_right", "joint_finger_tip_left"])

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

    color, depth = digits.render()
    digits.updateGUI(color, depth)

    conf_dict = {
        "cube" : {},
        # "banana" : {},
        # "apple" : {},
        # "baseball" : {},
        # "can" : {},
        # "bottle" : {}
    }

    object_id_list = []
    target_obj_name = "cube"
    target_obj_id = None
    create_objects(conf_dict)
    for objName in conf_dict :
        object_id_list.append(conf_dict[objName]["objId"])
        if target_obj_name == objName :
            target_obj_id = conf_dict[objName]["objId"]


    target_model_path = "models/grasp/cube/field['tactileColorL', 'tactileColorR', 'visionColor']_N120_i4.pth"
    model = Model(["tactileColorL", "tactileColorR", "visionColor"])
    device = torch.device("cuda:0")
    model.to(device)
    model.load(target_model_path)
    model.eval()

    environ = {}

    environ["rob"] = rob
    environ["robStartPos"] = rob.pos
    environ["cam"] = cam
    environ["digits"] = digits
    environ["sensorID"] = sensorID
    environ["target_obj_id"] = target_obj_id
    environ["object_id_list"] = object_id_list
    environ["transform"] = transform
    environ["robStartPos"] = [0.5, 0.0, 0.1]
    environ["robotID"] = robotID
    environ["conf_dict"] = conf_dict

    feasibility_socres = get_feasibility(environ, model)
    for score in feasibility_socres :
        print(score[0])

    pb.disconnect()  # Close PyBullet

if __name__ == '__main__':
    main()
