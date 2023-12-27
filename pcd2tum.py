from pyntcloud import PyntCloud
import os
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt

# Get the current file path
current_file_path = os.path.join(os.path.expanduser("~"),'lvi-sam','results')
print(current_file_path)
pcd_file_path = os.path.join(current_file_path,'transformations.pcd')
txt_file_path = os.path.join(current_file_path,'lvisam.txt')
# Read the PCD file
cloud = PyntCloud.from_file(pcd_file_path)

# Access the point cloud data
poses = []
for i in range(cloud.points.shape[0]):
    quad = Rotation.from_euler('xyz', [cloud.points.roll[i],cloud.points.pitch[i],cloud.points.yaw[i]], degrees=False).as_quat()
    pose = [cloud.points.time[i], cloud.points.x[i], cloud.points.y[i], cloud.points.z[i], quad[0], quad[1], quad[2], quad[3]]
    poses.append(pose)
# write in txt file
with open(txt_file_path, 'w') as f:
    for pose in poses:
        data = [str(pose[0]), str(pose[1]), str(pose[2]), str(pose[3]), str(pose[4]), str(pose[5]), str(pose[6]), str(pose[7])]
        f.write(' '.join(data) + '\n')
print("Done!")

