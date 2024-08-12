import pyrealsense2 as rs
from ultralytics import YOLO
import supervision as sv
import cv2
import yaml
import numpy as np
import math
import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class JengaBlockDetector(Node):
    def __init__(self):
        super().__init__('jenga_block_detector')

        # Load camera calibration data
        with open('newcalib_cam2.yaml', 'r') as f:
            dataDict = yaml.load(f, Loader=yaml.SafeLoader)
        self.camera_matrix = np.array(dataDict['camera_matrix'])
        self.dist_coeff = np.array(dataDict['dist_coeff'])
        self.rvecs = np.array(dataDict['rvecs'])
        self.tvecs = np.array(dataDict['tvecs'])

        # Initialize the ROS publisher
        self.publisher_ = self.create_publisher(Float32MultiArray, 'jenga_pose', 10)

        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device('747612060071')
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # Load the YOLO model
        self.model = YOLO('best.pt')

        # Set up the real-world parameters
        self.SF = 83  # scaling factor
        self.c_X_w = 20
        self.c_Y_w = 57
        self.c_Z_w = -83

        # Create the rotation matrix
        theta_z = math.pi
        self.R = np.array([[math.cos(theta_z), -math.sin(theta_z), 0],
                           [math.sin(math.pi), math.cos(math.pi), 0],
                           [0, 0, 1]])
        self.t = np.array([[self.c_X_w], [self.c_Y_w], [self.c_Z_w]])
        self.w_R_c = np.concatenate([self.R, self.t], 1)
        e = np.array([[0, 0, 0, 1]])
        self.w_R_c = np.concatenate([self.w_R_c, e])

    def world_coordinates(self, scale_factor, obb_xyxyxyxy):
        print(f"obb_xyxyxyxy shape: {obb_xyxyxyxy.shape}")

        # Convert pixel coordinates to camera coordinates
        def pixel_to_camera(p):
            P_c = np.matmul(np.linalg.inv(self.camera_matrix), p)
            return scale_factor * P_c

        p_lr = pixel_to_camera(np.array([obb_xyxyxyxy[0][0][0], obb_xyxyxyxy[0][0][1], 1]))
        p_ur = pixel_to_camera(np.array([obb_xyxyxyxy[0][1][0], obb_xyxyxyxy[0][1][1], 1]))
        p_ul = pixel_to_camera(np.array([obb_xyxyxyxy[0][2][0], obb_xyxyxyxy[0][2][1], 1]))
        p_ll = pixel_to_camera(np.array([obb_xyxyxyxy[0][3][0], obb_xyxyxyxy[0][3][1], 1]))

        # Convert camera coordinates to world coordinates
        def camera_to_world(P_c):
            P_wT = np.matmul(self.w_R_c, np.array([P_c[0], P_c[1], P_c[2], 1]))
            return np.array([P_wT[1], P_wT[0], -P_wT[2], P_wT[3]])

        P_w_lr = camera_to_world(p_lr)
        P_w_ur = camera_to_world(p_ur)
        P_w_ul = camera_to_world(p_ul)
        P_w_ll = camera_to_world(p_ll)

        print(P_w_lr)
        print(P_w_ur)
        print(P_w_ul)
        print(P_w_ll)

        # Calculate orientation and centroid
        x_bottom = (obb_xyxyxyxy[0][3][0] + obb_xyxyxyxy[0][0][0]) / 2
        y_bottom = (obb_xyxyxyxy[0][3][1] + obb_xyxyxyxy[0][0][1]) / 2
        x_top = (obb_xyxyxyxy[0][2][0] + obb_xyxyxyxy[0][1][0]) / 2
        y_top = (obb_xyxyxyxy[0][2][1] + obb_xyxyxyxy[0][1][1]) / 2
        
        # Calculate the orientation angle theta
        theta = math.atan2(y_top - y_bottom, x_top - x_bottom)
        theta_degreesT = math.degrees(theta)
        
        if theta_degreesT < 0:
            theta_degrees = -theta_degreesT
        else:
            theta_degrees = 180 - theta_degreesT

        print(theta)
        print(theta_degrees)
        
        yll = P_w_ll[1]
        ylr = P_w_lr[1]
        yul = P_w_ul[1]
        yur = P_w_ur[1]
        
        minim = yll
        minimx = P_w_ll[0]
        minimz = P_w_ll[2]

        if ylr < minim:
            minim = ylr
            minimx = P_w_lr[0]
            minimz = P_w_lr[2]
        if yul < minim:
            minim = yul
            minimx = P_w_ul[0]
            minimz = P_w_ul[2]
        if yur < minim:
            minim = yur
            minimx = P_w_ur[0]
            minimz = P_w_ur[2]
        
        l =  7.1
        cent_x = minimx + (l * math.sin(theta))
        cent_y = minim + (l * math.cos(theta))
        
        print(cent_x)
        print(cent_y)

        msg = Float32MultiArray()
        msg.data = [cent_x, cent_y, minimz, theta_degrees]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

        # Publish the calculated pose
     #   self.publish_pose(cent_x, cent_y, minimz, theta_degrees)

        return [P_w_lr, P_w_ur, P_w_ul, P_w_ll]

   # def publish_pose(self, cent_x, cent_y, minimz, theta_degrees):
        # Publish the Jenga block pose
    #    msg = Float32MultiArray()
     #   msg.data = [cent_x, cent_y, minimz, theta_degrees]
      #  self.publisher_.publish(msg)
       # self.get_logger().info(f'Publishing: {msg.data}')

    def display_coordinates(self, coordinates):
        # Display the coordinates in a separate window
        display_image = np.zeros((600, 800, 3), dtype=np.uint8)
        y0, dy = 30, 30
        for i, coord in enumerate(coordinates):
            text = f"Block {i+1} Coordinates:"
            y = y0 + i * dy * 5
            cv2.putText(display_image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            corner_names = ["Lower Right", "Upper Right", "Upper Left", "Lower Left"]
            for j, corner in enumerate(coord):
                corner_text = f" {corner_names[j]}: [{corner[0]:.2f}, {corner[1]:.2f}, {corner[2]:.2f}]"
                cv2.putText(display_image, corner_text, (10, y + (j + 1) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Coordinates', display_image)

    def generate_colors(self, num_colors):
        # Generate random colors for bounding boxes
        return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]

    def process_frames(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())
                results = self.model.predict(img, conf=0.79)

                coordinates = []
                if results[0].obb is not None:
                    obb_boxes = results[0].obb
                    colors = self.generate_colors(len(obb_boxes))
                    for i, obb in enumerate(obb_boxes):
                        obb_xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
                        print(f"obb_xyxyxyxy: {obb_xyxyxyxy}")
                        scale_factor = self.SF
                        corners = self.world_coordinates(scale_factor, obb_xyxyxyxy)
                        coordinates.append([corner[:3] for corner in corners])

                        color = colors[i]
                        for j in range(len(corners)):
                            p1 = (int(obb_xyxyxyxy[0][j][0]), int(obb_xyxyxyxy[0][j][1]))
                            p2 = (int(obb_xyxyxyxy[0][(j+1) % 4][0]), int(obb_xyxyxyxy[0][(j+1) % 4][1]))
                            cv2.line(img, p1, p2, color, 2)

                    self.display_coordinates(coordinates)

                cv2.imshow('RealSense', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"Error during rclpy shutdown: {e}")
            

def main(args=None):
    rclpy.init(args=args)
    node = JengaBlockDetector()
    try:
        rclpy.spin(node)  # Keep the node running to process callbacks
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
