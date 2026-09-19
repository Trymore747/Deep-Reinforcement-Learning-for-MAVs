import gz.transport13 as gz
import numpy as np
import time

node = gz.Node()

depth_msg = None

def cb(msg):
    global depth_msg
    depth_msg = msg

node.subscribe(
    "/gazebo/default/iris/depthcamera_link/camera/image",
    cb
)

print("Waiting for depth images...")

while depth_msg is None:
    time.sleep(0.1)

print("Depth image received!")

# Convert to numpy
width = depth_msg.width
height = depth_msg.height
data = np.array(depth_msg.data, dtype=np.float32)
depth = data.reshape(height, width)

print("Depth shape:", depth.shape)
print("Min depth:", np.min(depth))
print("Max depth:", np.max(depth))

