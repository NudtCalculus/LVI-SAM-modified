from pyntcloud import PyntCloud
import os
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt

def rotationError(pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5*(a+b+c-1.0)
    return np.arccos(max(min(d,1.0),-1.0))*180.0/np.pi

def translationError(pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx**2+dy**2+dz**2)


# Get the current file path
current_file_path = os.path.join(os.path.expanduser("~"),'lvi-sam','results')
print(current_file_path)
pcd_file_path = os.path.join(current_file_path,'transformations.pcd')
# txt_file_path = os.path.join(current_file_path,'transformations.txt')
# Read the PCD file
cloud = PyntCloud.from_file(pcd_file_path)

# Access the point cloud data
poses = []
trans = []
for i in range(cloud.points.shape[0]):
    R = Rotation.from_euler('xyz', [cloud.points.roll[i],cloud.points.pitch[i],cloud.points.yaw[i]], degrees=False).as_matrix()
    t = np.array([cloud.points.x[i], cloud.points.y[i], cloud.points.z[i]])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    pose = [cloud.points.time[i], T]
    poses.append(pose)
    trans.append(t)

# Do further processing with the point cloud data
# ...
gt_poses = []
gt_trans = []
gt_file = os.path.join(current_file_path,'gt.txt')
with open(gt_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = line.split()
        time = float(data[0])
        t = np.array([float(data[1]), float(data[2]), float(data[3])])
        R = Rotation.from_quat([float(data[4]), float(data[5]), float(data[6]), float(data[7])]).as_matrix()
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        pose = [time, T]
        gt_poses.append(pose)
        gt_trans.append(t)

# Do further processing with the point cloud data
# ...
kiss_poses = []
kiss_trans = []
kiss_file = os.path.join(current_file_path,'kissicp.txt')
with open(kiss_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = line.split()
        time = float(data[0])
        t = np.array([float(data[1]), float(data[2]), float(data[3])])
        R = Rotation.from_quat([float(data[4]), float(data[5]), float(data[6]), float(data[7])]).as_matrix()
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        pose = [time, T]
        kiss_poses.append(pose)
        kiss_trans.append(t)

# Do further processing with the point cloud data
# ...
fastlio_poses = []
fastlio_trans = []
fastlio_file = os.path.join(current_file_path,'fastlio2.txt')
with open(fastlio_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = line.split()
        time = float(data[0])
        t = np.array([float(data[1]), float(data[2]), float(data[3])])
        R = Rotation.from_quat([float(data[4]), float(data[5]), float(data[6]), float(data[7])]).as_matrix()
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        pose = [time, T]
        fastlio_poses.append(pose)
        fastlio_trans.append(t)

# compute rotational and translational errors, end-to-end pose error
print("LVI-SAM:")
first_frame = poses[0]
last_frame = poses[-1]
closest_first_frame = min(gt_poses, key=lambda pose: abs(pose[0] - first_frame[0]))
closest_last_frame = min(gt_poses, key=lambda pose: abs(pose[0] - last_frame[0]))
print('First frame time (s): ', first_frame[0])
print('Closest first frame time (s): ', closest_first_frame[0])
print('Last frame time (s): ', last_frame[0])
print('Closest last frame time (s): ', closest_last_frame[0])
pose_delta_gt = np.dot(np.linalg.inv(closest_first_frame[1]), closest_last_frame[1])
pose_delta_result = np.dot(np.linalg.inv(first_frame[1]), last_frame[1])
pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

r_err = rotationError(pose_error) #deg
t_err = translationError(pose_error) #m
print('Rotation Error (deg): ', r_err)
print('Translation Error (m): ', t_err)
print('*'*20)

# compute rotational and translational errors, end-to-end pose error
print('KISS-ICP:')
first_frame = kiss_poses[0]
last_frame = kiss_poses[-1]
closest_first_frame = min(gt_poses, key=lambda pose: abs(pose[0] - first_frame[0]))
closest_last_frame = min(gt_poses, key=lambda pose: abs(pose[0] - last_frame[0]))
print('First frame time (s): ', first_frame[0])
print('Closest first frame time (s): ', closest_first_frame[0])
print('Last frame time (s): ', last_frame[0])
print('Closest last frame time (s): ', closest_last_frame[0])
pose_delta_gt = np.dot(np.linalg.inv(closest_first_frame[1]), closest_last_frame[1])
pose_delta_result = np.dot(np.linalg.inv(first_frame[1]), last_frame[1])
pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

r_err = rotationError(pose_error) #deg
t_err = translationError(pose_error) #m
print('Rotation Error (deg): ', r_err)
print('Translation Error (m): ', t_err)
print('*'*20)

# compute rotational and translational errors, end-to-end pose error
print("Fastlio2:")
first_frame = fastlio_poses[0]
last_frame = fastlio_poses[-1]
closest_first_frame = min(gt_poses, key=lambda pose: abs(pose[0] - first_frame[0]))
closest_last_frame = min(gt_poses, key=lambda pose: abs(pose[0] - last_frame[0]))
print('First frame time (s): ', first_frame[0])
print('Closest first frame time (s): ', closest_first_frame[0])
print('Last frame time (s): ', last_frame[0])
print('Closest last frame time (s): ', closest_last_frame[0])
pose_delta_gt = np.dot(np.linalg.inv(closest_first_frame[1]), closest_last_frame[1])
pose_delta_result = np.dot(np.linalg.inv(first_frame[1]), last_frame[1])
pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

r_err = rotationError(pose_error) #deg
t_err = translationError(pose_error) #m
print('Rotation Error (deg): ', r_err)
print('Translation Error (m): ', t_err)
print('*'*20)

# Plot the trajectory
fig = plt.figure()

# Plot ground truth trajectory
trans = np.array(trans)
gt_trans = np.array(gt_trans)
kiss_trans = np.array(kiss_trans)
fastlio_trans = np.array(fastlio_trans)
plt.plot(gt_trans[:,0], gt_trans[:,1], label='Ground Truth')

# Plot estimated trajectory
plt.plot(trans[:,0], trans[:,1], label='LVI-SAM')

# Plot estimated trajectory
plt.plot(kiss_trans[:,0], kiss_trans[:,1], label='KISS-ICP')

# Plot estimated trajectory
plt.plot(fastlio_trans[:,0], fastlio_trans[:,1], label='FASTLIO2')

# Set the title
plt.title('Trajectory-xy')
plt.legend()

# Show the plot
plt.show()