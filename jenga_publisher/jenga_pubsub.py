from ultralytics import YOLO
import yaml
import numpy as np
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

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

        # Initialize the ROS subscriber
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        
        # Initialize the ROS publisher
        self.publisher_ = self.create_publisher(Float32MultiArray, 'jenga_pose', 10)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
                
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

        # initiliatize the msg
        # self.xyztheta
       
    def image_callback(self,msg):

        # Process the image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model.predict(cv_image, conf=0.79, verbose=False)

        # YOLO inference            
        if results[0].obb is not None:

            obb_boxes = results[0].obb

            # Print debugging vector if no jenga blocks are available
            if len(obb_boxes) == 0:
                self.get_logger().info('No jenga block found in the workspace')
                self.xyztheta = [1.234, 1.234, 1.234, 1.234] 

            # Calculate the jenga centroid and store in global variable self.xyztheta
            else:
                for obb in obb_boxes:
                    obb_xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
                    xc, yc, zc, thetac = self.world_coordinates(self.SF, obb_xyxyxyxy)
                    self.xyztheta = [xc, yc, zc, thetac]
                         
    def camera_to_world(p):
        P_wT = np.matmul(self.w_R_c, np.array([p[0], p[1], p[2], 1]))
        return np.array([P_wT[1], P_wT[0], -P_wT[2], P_wT[3]])
        
    def pixel_to_camera(scale_factor, p):
        P_c = np.matmul(np.linalg.inv(self.camera_matrix), p)
        return scale_factor * P_c
        
    def world_coordinates(self, scale_factor, obb_xyxyxyxy):

        # Convert pixel coordinates to camera coordinates
        p_lr = self.pixel_to_camera(scale_factor, np.array([obb_xyxyxyxy[0][0][0], obb_xyxyxyxy[0][0][1], 1]))
        p_ur = self.pixel_to_camera(scale_factor, np.array([obb_xyxyxyxy[0][1][0], obb_xyxyxyxy[0][1][1], 1]))
        p_ul = self.pixel_to_camera(scale_factor, np.array([obb_xyxyxyxy[0][2][0], obb_xyxyxyxy[0][2][1], 1]))
        p_ll = self.pixel_to_camera(scale_factor, np.array([obb_xyxyxyxy[0][3][0], obb_xyxyxyxy[0][3][1], 1]))

        # Convert camera coordinates to world coordinate
        P_w_lr = self.camera_to_world(p_lr)
        P_w_ur = self.camera_to_world(p_ur)
        P_w_ul = self.camera_to_world(p_ul)
        P_w_ll = self.camera_to_world(p_ll)

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

        return cent_x, cent_y, minimz, theta_degrees

        
    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = [self.xyztheta[0], self.xyztheta[1], self.xyztheta[2], self.xyztheta[3]]
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: {msg.data}')
            

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
