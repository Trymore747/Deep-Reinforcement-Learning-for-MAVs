import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

DEPTH_TOPIC = '/camera/depth/image_raw'  # change ONLY if needed

class DepthTest(Node):
    def __init__(self):
        super().__init__('depth_test')
        self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.cb,
            10
        )
        self.get_logger().info(f"Listening to {DEPTH_TOPIC}")

    def cb(self, msg):
        depth = np.frombuffer(msg.data, dtype=np.float32)
        depth = depth.reshape(msg.height, msg.width)

        front = depth[:, depth.shape[1]//2]
        min_front = np.nanmin(front)

        self.get_logger().info(
            f"Depth front min: {min_front:.2f} m"
        )

def main():
    rclpy.init()
    node = DepthTest()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

