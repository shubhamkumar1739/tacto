# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging

import hydra
import numpy as np
import pybullet as p

import pybulletX as px

import tacto

from sawyer_gripper import SawyerGripper

log = logging.getLogger(__name__)


def get_object_pose(objID):
    res = p.getBasePositionAndOrientation(objID)

    world_positions = res[0]
    world_orientations = res[1]

    objStartPos = [0.50, 0, 0.05]
    objStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])

    if (world_positions[0] ** 2 + world_positions[1] ** 2) > 0.8 ** 2:
        p.resetBasePositionAndOrientation(objID, objStartPos, objStartOrientation)
        return objStartPos, objStartOrientation

    world_positions = np.array(world_positions)
    world_orientations = np.array(world_orientations)

    return (world_positions, world_orientations)

# Load the config YAML file from examples/conf/grasp.yaml
@hydra.main(config_path="conf", config_name="grasp2")
def main(cfg):
    # Initialize digits
    digits = tacto.Sensor(**cfg.tacto)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    robot = SawyerGripper(**cfg.sawyer_gripper)

    # [21, 24]
    digits.add_camera(robot.id, robot.digit_links)

    # Add object to pybullet and digit simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj, True)

    # apple = px.Body(**cfg.apple)
    # digits.add_body(apple, True)

    # banana = px.Body(**cfg.banana)
    # digits.add_body(banana, True)

    baseball = px.Body(**cfg.baseball)
    digits.add_body(baseball, True)

    can = px.Body(**cfg.can)
    digits.add_body(can, True)

    bottle = px.Body(**cfg.bottle)
    digits.add_body(bottle, True)

    np.set_printoptions(suppress=True)

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    robot.reset()

    panel = px.gui.RobotControlPanel(robot)
    panel.start()

    while True:
        color, depth = digits.render()
        digits.updateGUI(color, depth)

        # px.simulationStep()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
