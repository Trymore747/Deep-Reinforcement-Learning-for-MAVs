#!/usr/bin/env python3
"""
A* Path Planner for Dynamic 3D Navigation
High-speed navigation using A* algorithm with dynamic obstacle avoidance
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial import KDTree
import heapq
from collections import defaultdict
import time


class AStarNode:
    """Node for A* search"""
    def __init__(self, position, g_cost=float('inf'), h_cost=0, parent=None):
        self.position = tuple(position)  # (x, y, z)
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic to goal
        self.f_cost = g_cost + h_cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __hash__(self):
        return hash(self.position)


class AStarPlanner:
    """A* path planner with dynamic obstacle avoidance for high-speed navigation"""
    
    def __init__(self):
        rospy.init_node('astar_planner', anonymous=True)
        
        # Parameters
        self.grid_resolution = rospy.get_param('~grid_resolution', 0.5)  # meters
        map_size_param = rospy.get_param('~map_size', [80, 40, 3])
        # Parse map_size if it's a string
        if isinstance(map_size_param, str):
            import ast
            self.map_size = ast.literal_eval(map_size_param)
        else:
            self.map_size = map_size_param
        self.robot_radius = rospy.get_param('~robot_radius', 0.5)
        self.obstacle_clearance = rospy.get_param('~obstacle_clearance', 0.8)
        self.dynamic_clearance = rospy.get_param('~dynamic_clearance', 1.0)
        self.max_speed = rospy.get_param('~max_speed', 3.0)
        self.replan_rate = rospy.get_param('~replan_rate', 2.0)  # Hz
        
        # State
        self.current_pos = None
        self.current_vel = None
        self.goal_pos = None
        self.obstacle_cloud = None
        self.path = []
        self.path_index = 0
        
        # Occupancy grid
        self.grid_origin = [-self.map_size[0]/2, -self.map_size[1]/2, 0]
        self.grid_shape = [int(self.map_size[i]/self.grid_resolution) for i in range(3)]
        self.occupancy_grid = np.zeros(self.grid_shape, dtype=np.uint8)
        
        # Dynamic obstacles tracking
        self.dynamic_obstacles = []
        self.obstacle_kdtree = None
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_pub = rospy.Publisher('/astar_path', PoseStamped, queue_size=100)
        
        # Subscribers
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/camera/depth/points', PointCloud2, self.pointcloud_callback)
        
        # Timing
        self.last_plan_time = rospy.Time.now()
        self.planning_interval = rospy.Duration(1.0 / self.replan_rate)
        
        rospy.loginfo("A* Planner initialized - High-speed navigation mode")
        rospy.loginfo(f"Grid resolution: {self.grid_resolution}m, Map size: {self.map_size}")
        rospy.loginfo(f"Max speed: {self.max_speed} m/s, Replan rate: {self.replan_rate} Hz")
    
    def odom_callback(self, msg):
        """Update current position and velocity"""
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
    
    def goal_callback(self, msg):
        """Receive new goal and trigger planning"""
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        rospy.loginfo(f"New goal received: {self.goal_pos}")
        self.path = []
        self.path_index = 0
        
        # Plan immediately
        rospy.loginfo("Starting path planning...")
        self.plan_path()
    
    def pointcloud_callback(self, msg):
        """Update obstacle map from point cloud"""
        points = []
        for point in pc2.read_points(msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])
        
        if len(points) > 0:
            self.obstacle_cloud = np.array(points)
            self.obstacle_kdtree = KDTree(self.obstacle_cloud)
            self.update_occupancy_grid()
    
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        try:
            grid_pos = [
                int((pos[i] - self.grid_origin[i]) / self.grid_resolution)
                for i in range(3)
            ]
            # Clamp to valid range
            grid_pos = [
                max(0, min(grid_pos[i], self.grid_shape[i] - 1))
                for i in range(3)
            ]
            return tuple(grid_pos)
        except (IndexError, TypeError, ValueError):
            rospy.logwarn(f"Invalid position for grid conversion: {pos}")
            return None
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        world_pos = [
            grid_pos[i] * self.grid_resolution + self.grid_origin[i]
            for i in range(3)
        ]
        return np.array(world_pos)
    
    def find_nearest_valid_grid(self, grid_pos):
        """Find nearest valid grid position to given position"""
        valid_pos = [
            max(0, min(grid_pos[i], self.grid_shape[i] - 1))
            for i in range(3)
        ]
        return tuple(valid_pos)
    
    def is_valid_grid(self, grid_pos):
        """Check if grid position is within bounds"""
        try:
            return all(0 <= grid_pos[i] < self.grid_shape[i] for i in range(3))
        except (IndexError, TypeError):
            return False
    
    def update_occupancy_grid(self):
        """Update occupancy grid from obstacle point cloud"""
        self.occupancy_grid.fill(0)
        
        if self.obstacle_cloud is None:
            return
        
        # Mark obstacles with clearance
        clearance_cells = int(self.obstacle_clearance / self.grid_resolution)
        
        for point in self.obstacle_cloud:
            grid_pos = self.world_to_grid(point)
            if not self.is_valid_grid(grid_pos):
                continue
            
            # Inflate obstacles
            for dx in range(-clearance_cells, clearance_cells + 1):
                for dy in range(-clearance_cells, clearance_cells + 1):
                    for dz in range(-1, 2):  # Less inflation in Z
                        nx, ny, nz = grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz
                        if self.is_valid_grid((nx, ny, nz)):
                            dist = np.sqrt(dx**2 + dy**2 + dz**2) * self.grid_resolution
                            if dist <= self.obstacle_clearance:
                                self.occupancy_grid[nx, ny, nz] = 1
    
    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def get_neighbors(self, grid_pos):
        """Get valid neighboring cells (26-connectivity)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    neighbor = (grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz)
                    
                    if self.is_valid_grid(neighbor) and self.occupancy_grid[neighbor] == 0:
                        neighbors.append(neighbor)
        
        return neighbors
    
    def plan_path(self):
        """Plan path using A* algorithm"""
        if self.current_pos is None or self.goal_pos is None:
            rospy.logwarn("Cannot plan: missing position or goal")
            return
        
        rospy.loginfo(f"Planning from {self.current_pos} to {self.goal_pos}")
        
        start_time = time.time()
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(self.current_pos)
        goal_grid = self.world_to_grid(self.goal_pos)
        
        if start_grid is None or goal_grid is None:
            rospy.logwarn("Invalid start or goal position")
            return
        
        rospy.loginfo(f"Grid start: {start_grid}, goal: {goal_grid}")
        
        if not self.is_valid_grid(start_grid) or not self.is_valid_grid(goal_grid):
            rospy.logwarn(f"Start {start_grid} or goal {goal_grid} outside valid grid")
            # Try to find nearest valid goal
            goal_grid = self.find_nearest_valid_grid(goal_grid)
            if goal_grid is None:
                rospy.logerr("Cannot find valid goal position")
                return
            rospy.loginfo(f"Using adjusted goal: {goal_grid}")
        
        # A* search
        open_set = []
        start_node = AStarNode(start_grid, g_cost=0, h_cost=self.heuristic(start_grid, goal_grid))
        heapq.heappush(open_set, start_node)
        
        closed_set = set()
        g_scores = {start_grid: 0}
        
        iterations = 0
        max_iterations = 10000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            
            if current.position in closed_set:
                continue
            
            # Goal reached
            if self.heuristic(current.position, goal_grid) < 1.5:
                self.reconstruct_path(current)
                elapsed = time.time() - start_time
                rospy.loginfo(f"Path found! Length: {len(self.path)} points, Time: {elapsed:.3f}s, Iterations: {iterations}")
                self.path_index = 0
                return
            
            closed_set.add(current.position)
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate cost
                move_cost = self.heuristic(current.position, neighbor_pos)
                tentative_g = current.g_cost + move_cost
                
                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = tentative_g
                    h_cost = self.heuristic(neighbor_pos, goal_grid)
                    neighbor_node = AStarNode(neighbor_pos, g_cost=tentative_g, h_cost=h_cost, parent=current)
                    heapq.heappush(open_set, neighbor_node)
        
        rospy.logwarn(f"No path found after {iterations} iterations")
    
    def reconstruct_path(self, node):
        """Reconstruct path from goal to start"""
        path = []
        current = node
        while current is not None:
            path.append(self.grid_to_world(current.position))
            current = current.parent
        
        self.path = path[::-1]  # Reverse to get start->goal
        
        # Publish path for visualization
        for point in self.path:
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.pose.position.x = point[0]
            msg.pose.position.y = point[1]
            msg.pose.position.z = point[2]
            self.path_pub.publish(msg)
    
    def follow_path(self):
        """Generate velocity commands to follow path"""
        if len(self.path) == 0 or self.current_pos is None:
            # No path - try planning if we have a goal
            if self.goal_pos is not None:
                rospy.logwarn_throttle(2.0, "No path available, attempting to plan...")
                self.plan_path()
            return
        
        # Check if we need to replan
        current_time = rospy.Time.now()
        if (current_time - self.last_plan_time) > self.planning_interval and self.goal_pos is not None:
            self.last_plan_time = current_time
            self.plan_path()
        
        # Find current target point on path
        while self.path_index < len(self.path) - 1:
            target = self.path[self.path_index]
            dist = np.linalg.norm(target - self.current_pos)
            if dist < self.grid_resolution * 1.5:
                self.path_index += 1
            else:
                break
        
        if self.path_index >= len(self.path):
            # Reached goal
            rospy.loginfo_throttle(1.0, "Goal reached!")
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Calculate desired velocity
        target = self.path[self.path_index]
        direction = target - self.current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.01:
            direction = direction / distance
            
            # Speed control based on distance to goal
            goal_dist = np.linalg.norm(self.goal_pos - self.current_pos)
            if goal_dist < 3.0:
                speed = min(self.max_speed * 0.5, goal_dist * 0.5)
            else:
                speed = self.max_speed
            
            velocity = direction * speed
            
            # Publish command
            cmd = Twist()
            cmd.linear.x = velocity[0]
            cmd.linear.y = velocity[1]
            cmd.linear.z = velocity[2]
            self.cmd_vel_pub.publish(cmd)
            
            # Debug output
            rospy.loginfo_throttle(1.0, f"Following path: pos={self.current_pos[0]:.1f}m, target={target[0]:.1f}m, speed={speed:.1f}m/s")
    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(20)  # 20 Hz control
        
        while not rospy.is_shutdown():
            self.follow_path()
            rate.sleep()


if __name__ == '__main__':
    try:
        planner = AStarPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
