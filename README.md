Need ROS2 to run this jenga detector and OBB publisher. Clone it to the ROS2 workspace and build using colcon tools. 

Commands to launch the realsense camera and the publisher:
In one terminal, clone all your ROS2 workspaces which are relevant for the YOLO pacakge to run and then
  $ ros2 launch realsense2_camera rs_launch.py
In a second terminal, navigate to the package location for example I have it here: (Please note that I have overlayed this ws as well)
  $ cd ~/yolo_colcon_ws/src/jenga_publisher/jenga_publisher
  Then run:
  $ python3 jenga_pubsub.py
  
