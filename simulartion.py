import pybullet as p
import pybullet_data
import time
import random
import math
from enum import Enum

# Simulation speed
SIM_HZ = 240
DT = 1/SIM_HZ

# Number of cubes spawned on the table
NUM_CUBES = 30

# Robot forces
ARM_FORCE = 2000
GRIP_FORCE = 600

# Heights used for safe movement
LIFT_HEIGHT = 0.9
APPROACH = 0.12
GRASP_OFFSET = 0.01

# Cube size
CUBE_SIZE = 0.03

# Colors that should be sorted
TARGET_COLORS = ["RED","GREEN","BLUE"]

# Color definitions
COLOR_TABLE = {
"RED":[0.9,0.1,0.1,1],
"GREEN":[0.1,0.8,0.2,1],
"BLUE":[0.1,0.2,0.9,1],
"YELLOW":[0.9,0.9,0.1,1],
"PURPLE":[0.7,0.2,0.8,1],
"CYAN":[0.1,0.9,0.9,1]
}

# Bin positions where cubes are dropped
DROP_ZONES = {
"RED":[0.40,0.60,0.63],
"GREEN":[1.00,0.60,0.63],
"BLUE":[0.70,-0.60,0.63]
}

# Panda robot joint limits
LOWER=[-2.9,-1.76,-2.9,-3,-2.9,-0.01,-2.9]
UPPER=[2.9,1.76,2.9,-0.05,2.9,3.75,2.9]
JR=[5.8,3.52,5.8,3,5.8,3.76,5.8]

# Comfortable starting pose
REST=[0,-0.6,0,-2.3,0,1.8,0.8]

# End effector facing downward
DOWN=p.getQuaternionFromEuler([math.pi,0,0])

# Metrics for performance evaluation
pick_attempts=0
grasp_success=0
place_success=0
target_cubes=0


# States used in the pick-and-place sequence
class TaskState(Enum):
    IDLE=0
    SCAN=1
    APPROACH=2
    GRASP=3
    MOVE_TO_BIN=4
    RELEASE=5


# ----- Environment setup -----

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,1)

p.resetDebugVisualizerCamera(
1.6,45,-35,[0.75,0,0.65]
)

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf",[0.75,0,-0.65],useFixedBase=True)

robot=p.loadURDF("franka_panda/panda.urdf",[0,0,0],useFixedBase=True)

for i in range(7):
    p.resetJointState(robot,i,REST[i])


# Robot controller
class RobotAgent:

    def __init__(self,robot_id):
        self.robot=robot_id
        self.ee=11

    def step(self):
        p.stepSimulation()
        time.sleep(DT)

    # Move end effector smoothly using IK
    def move_to(self,pos,duration=0.55):

        start=p.getLinkState(self.robot,self.ee)[0]
        steps=int(duration/DT)

        for i in range(steps):

            t=i/steps
            s=3*t*t-2*t*t*t

            interp=[start[j]+s*(pos[j]-start[j]) for j in range(3)]

            joints=p.calculateInverseKinematics(
            self.robot,self.ee,interp,DOWN,
            lowerLimits=LOWER,
            upperLimits=UPPER,
            jointRanges=JR,
            restPoses=REST)

            for j in range(7):
                p.setJointMotorControl2(
                self.robot,j,p.POSITION_CONTROL,
                joints[j],force=ARM_FORCE)

            self.step()

    # Open gripper fingers
    def open_gripper(self):

        p.setJointMotorControl2(self.robot,9,p.POSITION_CONTROL,0.04,force=GRIP_FORCE)
        p.setJointMotorControl2(self.robot,10,p.POSITION_CONTROL,0.04,force=GRIP_FORCE)

        for _ in range(50):
            self.step()

    # Close gripper
    def close_gripper(self):

        p.setJointMotorControl2(self.robot,9,p.POSITION_CONTROL,0,force=GRIP_FORCE)
        p.setJointMotorControl2(self.robot,10,p.POSITION_CONTROL,0,force=GRIP_FORCE)

        for _ in range(70):
            self.step()

    # Check if cube is touching the gripper
    def verify_grasp(self,cube):

        contacts=p.getContactPoints(self.robot,cube)
        return len(contacts)>0


agent=RobotAgent(robot)


# Create a colored cube in the scene
def create_cube(pos,color):

    col=p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[CUBE_SIZE]*3)

    vis=p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[CUBE_SIZE]*3,
    rgbaColor=color)

    cube=p.createMultiBody(
    baseMass=0.35,
    baseCollisionShapeIndex=col,
    baseVisualShapeIndex=vis,
    basePosition=pos)

    return cube


cubes=[]

# Spawn cubes randomly on the table
for i in range(NUM_CUBES):

    x=random.uniform(0.55,0.95)
    y=random.uniform(-0.35,0.35)

    cname=random.choice(list(COLOR_TABLE.keys()))

    cubes.append({
    "id":create_cube([x,y,0.63],COLOR_TABLE[cname]),
    "name":cname
    })

# Let cubes settle on the table
for _ in range(200):
    p.stepSimulation()


# ----- Sorting process -----

state=TaskState.SCAN

for cube in cubes:

    if cube["name"] not in TARGET_COLORS:
        continue

    target_cubes+=1

    cube_id=cube["id"]
    pos,_=p.getBasePositionAndOrientation(cube_id)

    p.addUserDebugText(
    f"STATE: {state.name}",
    [0.6,0.3,1],
    textSize=1.5,
    lifeTime=0.1)

    state=TaskState.APPROACH

    agent.open_gripper()

    agent.move_to([pos[0],pos[1],LIFT_HEIGHT])
    agent.move_to([pos[0],pos[1],pos[2]+APPROACH])
    agent.move_to([pos[0],pos[1],pos[2]+GRASP_OFFSET])

    state=TaskState.GRASP

    agent.close_gripper()

    pick_attempts+=1

    if not agent.verify_grasp(cube_id):
        continue

    grasp_success+=1

    constraint=p.createConstraint(
    robot,11,cube_id,-1,
    p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])

    state=TaskState.MOVE_TO_BIN

    drop=DROP_ZONES[cube["name"]]

    agent.move_to([pos[0],pos[1],LIFT_HEIGHT])
    agent.move_to([drop[0],drop[1],LIFT_HEIGHT])
    agent.move_to([drop[0],drop[1],drop[2]+0.02])

    state=TaskState.RELEASE

    agent.open_gripper()

    p.removeConstraint(constraint)

    agent.move_to([drop[0],drop[1],LIFT_HEIGHT])

    place_success+=1


# ----- Performance summary -----

print("\n==============================")
print("SORTING METRICS")
print("==============================")
print("Total Cubes:",NUM_CUBES)
print("Target Cubes:",target_cubes)
print("Pick Attempts:",pick_attempts)
print("Successful Grasps:",grasp_success)
print("Successful Placements:",place_success)

if pick_attempts>0:
    print("Grasp Success Rate:",
    round(grasp_success/pick_attempts*100,2),"%")

if target_cubes>0:
    print("Sorting Accuracy:",
    round(place_success/grasp_success*100,2),"%")

print("==============================")


# Keep simulation running
while True:
    p.stepSimulation()
    time.sleep(DT)
