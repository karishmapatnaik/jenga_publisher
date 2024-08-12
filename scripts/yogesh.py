import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry, VehicleStatus, SensorCombined
from squeeze_custom_msgs.msg import CollisionStatus
from std_msgs.msg import Float64MultiArray
import math
import time
import numpy as np

class AdmittanceCollisionDetection(Node):
    "Node for controlling a vehicle in offboard mode using position-velocity setpoints and restarting the mission on collision"

    def __init__(self) -> None: 
        super().__init__('admittance_collision_detection')

        # create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/offboard_control_mode/in', 10)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/trajectory_setpoint/in', 10)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/vehicle_command/in', 10)
        self.collision_status_publisher = self.create_publisher(
            CollisionStatus, '/collision_status', 10)
       
        self.collision_quat_publisher = self.create_publisher(
            Float64MultiArray, '/mission_data/collision_quat', 10)
        self.collision_acc_publisher = self.create_publisher(
            Float64MultiArray, '/mission_data/collision_acc', 10)
        self.delta_wp_publisher = self.create_publisher(Float64MultiArray, '/mission_data/delta_wp', 10)
        self.collision_timer_publisher = self.create_publisher(Float64MultiArray, '/mission_data/collision_timer', 10)
        
        # create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, 'fmu/vehicle_odometry/out',self.vehicle_odometry_callback, 10)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/vehicle_status/out', self.vehicle_status_callback, 10)
        self.sensor_combined_subscriber = self.create_subscription(
            SensorCombined, '/fmu/sensor_combined/out', self.sensor_combined_callback, 10)
        
        # create a timer_callback to publish control commands
        self.timer = self.create_timer(0.01, self.timer_callback)

        #------------------------

        # initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.sensor_combined = SensorCombined()
        self.collision_status = False
        self.t_initial = self.get_clock().now().nanoseconds / 10**9
        self.delta_t = 0
        self.thres_delta_t = 60
        self.yaw = 0.0 # 0 -> Aligned || np.pi/4 -> Non-Aligned 
        self.waypoints = self.generateWaypoints()
        self.Collision_Acc_x = 0 
        self.Collision_Acc_y = 0
        self.Collision_Acc_z = 0
        self.wp_num = 0
        self.thres_error = .15
        self.max_collision = 1
        self.collision_count = 0
        self.thresh_acc = 15.0
        self.setpoint_scale = .03

        # collision_timer
        self.collision_timer_i = self.get_clock().now().nanoseconds/10**9
        self.collision_timer_thres = 0.5
        self.collision_timer =  0
        
        
        # Arming commands -------CHECK 1------------
        while self.offboard_setpoint_counter < 10:
            #self.engage_offboard_mode()
            self.arm()
            self.offboard_setpoint_counter +=1
            time.sleep(0.1)
            
    def generateWaypoints(self):
        self.get_logger().info("----Generating Waypoints----")
        wp = []
        #-------CHECK 2------------
        # Wp 0
        wp.append([0.35 , 0.39, -0.8, 0.0 , 0.0,float("nan"), self.yaw])
        # Wp 1
        wp.append([0.35 , 0.39, -0.8, 0.0 , 0.0,float("nan"), self.yaw])
        # Wp 2
        wp.append([0.35 , 0.39, -0.8, 0.0 , 0.0,float("nan"), self.yaw])
        # Wp 3 (modified after collision 1)
        wp.append([0.35 , 0.39, -0.8, float("nan"),float("nan"),float("nan"), self.yaw])  # Velocities depend on angle of collision and ADD Yaw
        # Wp 4 (modified after collision 2)
        wp.append([0.35 , 0.39, -0.8, float("nan"),float("nan"),float("nan"), self.yaw])
        # Wp 5 (hovering over landing wp)
        wp.append([0.0 , 0.0, -0.8, float("nan"),float("nan"),float("nan"), self.yaw])
        # Wp 6 (landing wp)
        wp.append([0.0 , 0.0, 0.0, float("nan"),float("nan"),float("nan"), self.yaw])

        self.get_logger().info("----Waypoint generation completed----")
        return wp
        
    def vehicle_odometry_callback(self, vehicle_odometry):
        self.vehicle_odometry = vehicle_odometry

    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status

    def sensor_combined_callback(self, sensor_combined):
        self.sensor_combined = sensor_combined 
        collision_timer_msg = Float64MultiArray()
        collision_timer_msg.data = [float(self.collision_timer_i), float(self.collision_timer_thres), float(self.collision_timer)]
        self.collision_timer_publisher.publish(collision_timer_msg)
        self.checkCollision()
                 
    def arm(self):            
        """send an arm command to the vehicle"""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1 = 1.0, param2=6.0)
        self.get_logger().info('Switching to offboard mode!')

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_collision_status(self):
        msg = CollisionStatus()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.collision_status = self.collision_status
        self.collision_status_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def wp_publisher(self, wp_num):
        # add different cases to publish waypoints
        msg = TrajectorySetpoint()
        msg.x = self.waypoints[self.wp_num][0]
        msg.y = self.waypoints[self.wp_num][1]
        msg.z = self.waypoints[self.wp_num][2]
        msg.vx = self.waypoints[self.wp_num][3]
        msg.vy =  self.waypoints[self.wp_num][4]
        msg.vz =  self.waypoints[self.wp_num][5]
        msg.yaw = self.waypoints[self.wp_num][6] # 0 degree
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[msg.x, msg.y, msg.z]}")

    def check(self):        

        if self.wp_num == 0:
            if self.euclideanError(self.wp_num) >= self.thres_error:
                self.wp_publisher(self.wp_num)
            else:        
                self.wp_num += 1
                self.wp_publisher(self.wp_num) 

        if self.wp_num == 1:
            if self.euclideanError(self.wp_num) >= self.thres_error:
                self.wp_publisher(self.wp_num)
            else:        
                self.wp_num += 1
                self.t_initial = self.get_clock().now().nanoseconds / 10**9
                self.wp_publisher(self.wp_num)
        
        if self.wp_num == 2:
            #-------CHECK 3------------
            if not self.collision_status and self.delta_t <= self.thres_delta_t:
                self.wp_publisher(self.wp_num)
                self.get_logger().info(f"Ready to take hit!: Time left(sec)-{self.delta_t-self.thres_delta_t}, wp num: {self.wp_num}")
            elif not self.collision_status and self.delta_t > self.thres_delta_t:
                self.wp_num = 5 #hover over landing wp 
                self.wp_publisher(self.wp_num)
            elif self.collision_status and self.delta_t <= self.thres_delta_t:             
                # Modify next wp using AdmittanceController, Acc_x & Acc_y using some proportional function
                self.modifyWaypoint(self.wp_num)  #modification stays in admittance
                self.collision_status = False
                self.t_initial = self.get_clock().now().nanoseconds / 10**9
                self.wp_num = 2 # stays at this wp
                self.wp_publisher(self.wp_num)  
        
        if self.wp_num == 3:
            
            if not self.collision_status and self.delta_t <= self.thres_delta_t:
                self.wp_publisher(self.wp_num)
                self.get_logger().info(f"For second hit!: Time left-{self.delta_t-self.thres_delta_t}, wp num: {self.wp_num}")
            elif not self.collision_status and self.delta_t > self.thres_delta_t:
                self.wp_num = 5
                self.wp_publisher(self.wp_num)
            elif self.collision_status and self.delta_t <= self.thres_delta_t:             
                # Modify wp using AdmittanceController, Acc_x & Acc_y using some proportional function
                self.modifyWaypoint(self.wp_num + 1)  
                self.collision_status = False
                self.t_initial = self.get_clock().now().nanoseconds / 10**9
                self.wp_num += 1  
                self.wp_publisher(self.wp_num)           

        if self.wp_num == 4:
            if self.euclideanError(self.wp_num) >= self.thres_error:
                self.wp_publisher(self.wp_num)
            else:        
                self.wp_num = 4 # keep it hovering over the modified waypoint after collision 2
                self.get_logger().info(f"Hovering over modified wp!!, wp num: {self.wp_num}")
                self.wp_publisher(self.wp_num)
        
        if self.wp_num == 5: #hover over landing wp
            if self.euclideanError(self.wp_num) >= self.thres_error:
                self.wp_publisher(self.wp_num)
            else:        
                self.wp_num += 1
                self.wp_publisher(self.wp_num)

        if self.wp_num == 6: # land
            if self.euclideanError(self.wp_num) >= self.thres_error:
                self.wp_publisher(self.wp_num)
            else:        
                self.land()
                
        return None

def main(args = None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    admittance_collision_detection = AdmittanceCollisionDetection()
    rclpy.spin(admittance_collision_detection)
    admittance_collision_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)