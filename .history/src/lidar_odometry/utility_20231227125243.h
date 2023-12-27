#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/keypoints/harris_3d.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <unordered_map>

using namespace std;

typedef pcl::PointXYZI PointType;
typedef pcl::PointXYZINormal Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointCloud<Point>::ConstPtr PointCloudConstPtr;
typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::PointCloud<pcl::FPFHSignature33> Features;

struct NormalComputeParams {
  int search_method = 0;
  // search_method = 0: use KNN, = 1: use radius
  int k = 10;
  double radius = 1.0;
  int num_threads = 4;
};

struct HarrisParams {
  double harris_threshold_;
  bool harris_suppression_;
  double harris_radius_;
  bool harris_refine_;
  int harris_response_;
};

HarrisParams harris_params_;
NormalComputeParams normal_compute_params_;

class ParamServer
{
public:

    ros::NodeHandle nh;

    std::string PROJECT_NAME;

    std::string robot_id;

    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Velodyne Sensor Configuration: Velodyne
    int N_SCAN;
    int Horizon_SCAN;
    string timeField;
    int downsampleRate;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
   

#if IF_OFFICIAL
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;     //; R_lidar_imu, 即IMU -> LiDAR的旋转
    Eigen::Matrix3d extRPY; 
    Eigen::Vector3d extTrans;   //; t_lidar_imu, 即IMU -> LiDAR的平移
    Eigen::Quaterniond extQRPY;
#else
    static bool if_print_param;
    vector<double> R_imu_lidar_V;
    vector<double> t_imu_lidar_V;
    Eigen::Matrix3d R_imu_lidar;   //; R_imu_lidar, 即LiDAR -> IMU的旋转
    Eigen::Matrix3d R_lidar_imu;   //; R_imu_lidar.transpose()
    Eigen::Vector3d t_imu_lidar;   //; t_imu_lidar, 即LiDAR -> IMU的平移
    Eigen::Quaterniond Q_quat_lidar; //; R_quat_lidar, 即LiDAR -> IMU的四元数坐标系的旋转
#endif

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool loopClosureEnableFlag;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    // Point cloud registration params
    int icp_threads_;
    double sac_features_radius_;
    int loopclosure_method_;
    int registration_method_;

    ParamServer()
    {
        ROS_INFO("Start ParamServer.");
        nh.param<std::string>("/PROJECT_NAME", PROJECT_NAME, "sam");

        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>(PROJECT_NAME + "/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>(PROJECT_NAME + "/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>(PROJECT_NAME + "/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>(PROJECT_NAME + "/gpsTopic", gpsTopic, "odometry/gps");

        nh.param<bool>(PROJECT_NAME + "/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>(PROJECT_NAME + "/useGpsElevation", useGpsElevation, false);
        nh.param<float>(PROJECT_NAME + "/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>(PROJECT_NAME + "/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>(PROJECT_NAME + "/savePCD", savePCD, false);
        nh.param<std::string>(PROJECT_NAME + "/savePCDDirectory", savePCDDirectory, "/tmp/loam/");

        nh.param<int>(PROJECT_NAME + "/N_SCAN", N_SCAN, 16);
        nh.param<int>(PROJECT_NAME + "/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<std::string>(PROJECT_NAME + "/timeField", timeField, "time");
        nh.param<int>(PROJECT_NAME + "/downsampleRate", downsampleRate, 1);

        nh.param<float>(PROJECT_NAME + "/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>(PROJECT_NAME + "/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>(PROJECT_NAME + "/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>(PROJECT_NAME + "/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>(PROJECT_NAME + "/imuGravity", imuGravity, 9.80511);

    #if IF_OFFICIAL
        ROS_INFO("ParamServer official.");
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);
    #else
        ROS_INFO("ParamServer non-official.");
        //? mod: 修改外参读取方式
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicTranslation", t_imu_lidar_V, vector<double>());
        ROS_INFO("ParamServer non-official 11111.");
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicRotation", R_imu_lidar_V, vector<double>());
        ROS_INFO("ParamServer non-official 222222.");
        t_imu_lidar = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(t_imu_lidar_V.data(), 3, 1);
        ROS_INFO("ParamServer non-official 333333.");
        // Eigen::Matrix3d R_tmp;
        Eigen::Matrix3d R_tmp = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(R_imu_lidar_V.data(), 3, 3);
        ROS_INFO("ParamServer non-official 1st assert.");
        // ROS_ASSERT(abs(R_tmp.determinant()) > 0.9);   // 防止配置文件中写错，这里加一个断言判断一下
        ROS_INFO("ParamServer non-official 1st assert done.");
        R_imu_lidar = Eigen::Quaterniond(R_tmp).normalized().toRotationMatrix();
        R_lidar_imu = R_imu_lidar.transpose();

        //; yaw/pitch/roll的欧拉角绕着哪个轴逆时针旋转，结果为正数。一般来说是绕着+z、+y、+x
        std::string yaw_axis, pitch_axis, roll_axis;   
        nh.param<std::string>(PROJECT_NAME + "/yawAxis", yaw_axis, "+z");
        ROS_ASSERT(yaw_axis[0] == '+' || yaw_axis[0] == '-');
        nh.param<std::string>(PROJECT_NAME + "/pitchAxis", pitch_axis, "+y");
        ROS_ASSERT(pitch_axis[0] == '+' || pitch_axis[0] == '-');
        nh.param<std::string>(PROJECT_NAME + "/rollAxis", roll_axis, "+x");
        ROS_ASSERT(roll_axis[0] == '+' || roll_axis[0] == '-');
        ROS_ASSERT(yaw_axis[1] != pitch_axis[1] && yaw_axis[1] != roll_axis[1] && pitch_axis[1] != roll_axis[1]);

        //; 旋转的欧拉角坐标系(quat) -> IMU角速度、加速度坐标系(imu) 的旋转
        Eigen::Matrix3d R_imu_quat;   
        std::unordered_map<std::string, Eigen::Vector3d> col_map;
        col_map.insert({"+x", Eigen::Vector3d( 1,  0,  0)}); 
        col_map.insert({"-x", Eigen::Vector3d(-1,  0,  0)});
        col_map.insert({"+y", Eigen::Vector3d( 0,  1,  0)}); 
        col_map.insert({"-y", Eigen::Vector3d( 0, -1,  0)});
        col_map.insert({"+z", Eigen::Vector3d( 0,  0,  1)}); 
        col_map.insert({"-z", Eigen::Vector3d( 0,  0, -1)});
        R_imu_quat.col(2) = col_map[yaw_axis];
        R_imu_quat.col(1) = col_map[pitch_axis];
        R_imu_quat.col(0) = col_map[roll_axis];
        ROS_ASSERT(abs(R_imu_quat.determinant()) > 0.9);  

        ROS_INFO("ParamServer non-official  1111.");
        //; R_quat_lidar = R_quat_imu * R_imu_lidar
        Eigen::Matrix3d R_quat_lidar = R_imu_quat.transpose() * R_imu_lidar;  
        Q_quat_lidar = Eigen::Quaterniond(R_quat_lidar).normalized();

        if(if_print_param)
        {
            if_print_param = false;
            ROS_WARN_STREAM("=== R_imu_lidar : ===============");
            std::cout << R_imu_lidar << std::endl;
            ROS_WARN_STREAM("=== t_imu_lidar : ===============");
            std::cout << t_imu_lidar << std::endl;

            ROS_WARN_STREAM("=== R_imu_quat  : ===============");
            std::cout << "yawAxis = " << yaw_axis << ", col_map: " << col_map[yaw_axis].transpose()
                << ", pitchAxis = " << pitch_axis << ", col_map: " << col_map[pitch_axis].transpose()
                << ", rollAxis = " << roll_axis << ", col_map: " << col_map[roll_axis].transpose()
                << std::endl;
            std::cout << R_imu_quat << std::endl;

            ROS_WARN_STREAM("=== R_quat_lidar  : ===============");
            std::cout << R_quat_lidar << std::endl;
        }
    #endif

        ROS_INFO("ParamServer first endif.");

        nh.param<float>(PROJECT_NAME + "/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>(PROJECT_NAME + "/surfThreshold", surfThreshold, 0.1);
        nh.param<int>(PROJECT_NAME + "/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>(PROJECT_NAME + "/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>(PROJECT_NAME + "/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>(PROJECT_NAME + "/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>(PROJECT_NAME + "/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>(PROJECT_NAME + "/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>(PROJECT_NAME + "/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>(PROJECT_NAME + "/numberOfCores", numberOfCores, 2);
        nh.param<double>(PROJECT_NAME + "/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>(PROJECT_NAME + "/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>(PROJECT_NAME + "/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>(PROJECT_NAME + "/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>(PROJECT_NAME + "/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>(PROJECT_NAME + "/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<int>(PROJECT_NAME + "/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>(PROJECT_NAME + "/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>(PROJECT_NAME + "/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>(PROJECT_NAME + "/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>(PROJECT_NAME + "/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        nh.param<double>(PROJECT_NAME + "/harris_threshold", harris_params_.harris_threshold_, 0.000000005);
        nh.param<bool>(PROJECT_NAME + "/harris_suppression", harris_params_.harris_suppression_, true);
        nh.param<double>(PROJECT_NAME + "/harris_radius", harris_params_.harris_radius_, 0.1);
        nh.param<bool>(PROJECT_NAME + "/harris_refine", harris_params_.harris_refine_, false);
        nh.param<int>(PROJECT_NAME + "/harris_response", harris_params_.harris_response_, 1);

        nh.param<int>(PROJECT_NAME + "/icp_threads", icp_threads_, 8);
        nh.param<double>(PROJECT_NAME + "/feature_radius", sac_features_radius_, 2.0);
        nh.param<int>(PROJECT_NAME + "/loopclosure_method", loopclosure_method_, 0);
        nh.param<int>(PROJECT_NAME + "/registration_method", registration_method_, 1);

        usleep(100);
    }

    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    #if IF_OFFICIAL
        acc = extRot * acc;
    #else
        acc = R_lidar_imu * acc;
    #endif
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    #if IF_OFFICIAL
        gyr = extRot * gyr;
    #else
        gyr = R_lidar_imu * gyr;
    #endif
        
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);  
    #if IF_OFFICIAL
        Eigen::Quaterniond q_final = q_from * extQRPY;
    #else
        Eigen::Quaterniond q_final = q_from * Q_quat_lidar;
    #endif  
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        return imu_out;
    }
};

#if IF_OFFICIAL
#else
bool ParamServer::if_print_param = true;
#endif

template<typename T>
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, T thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

#endif

template <typename pointT>
void ComputeNormals(const typename pcl::PointCloud<pointT>::ConstPtr& input,
                    const NormalComputeParams& params,
                    Normals::Ptr normals) {
  typename pcl::search::KdTree<pointT>::Ptr search_method(
      new pcl::search::KdTree<pointT>);
  typename pcl::NormalEstimationOMP<pointT, pcl::Normal> norm_est;

  if (params.search_method == 0) {
    norm_est.setRadiusSearch(0);
    norm_est.setKSearch(params.k);
  } else if (params.search_method == 1) {
    norm_est.setKSearch(0);
    norm_est.setRadiusSearch(params.radius);
  } else {
    ROS_ERROR(
        "Wrong normal search method in lamp basestation normal computation. ");
    EXIT_FAILURE;
  }
  norm_est.setNumberOfThreads(params.num_threads);

  norm_est.setInputCloud(input);
  norm_est.setSearchMethod(search_method);
  norm_est.compute(*normals);
}

void ExtractNormals(const PointCloudConstPtr& input,
                                            Normals::Ptr normals,
                                            const NormalComputeParams& params) {
  normals->resize(input->size());
  if (input->size() == 0)
    return;
  // Check that there are normals to extract
  if (input->points[0].normal_x == 0 && input->points[0].normal_y == 0 &&
      input->points[0].normal_z == 0) {
    return ComputeNormals<Point>(input, params, normals);
  }
  int enable_omp = (1 < params.num_threads);
#pragma omp parallel for schedule(dynamic, 1) if (enable_omp)
  for (size_t i = 0; i < input->size(); ++i) {
    normals->points[i].normal_x = input->points[i].normal_x;
    normals->points[i].normal_y = input->points[i].normal_y;
    normals->points[i].normal_z = input->points[i].normal_z;
  }
}

void ComputeKeypoints(const PointCloudConstPtr& source,
                                                    const Normals::ConstPtr& source_normals,
                                                    const HarrisParams& params,
                                                    const int& num_threads,
                                                    PointCloud::Ptr source_keypoints) {
  pcl::HarrisKeypoint3D<Point, Point> harris_detector;

  harris_detector.setNonMaxSupression(params.harris_suppression_);
  harris_detector.setRefine(params.harris_refine_);
  harris_detector.setInputCloud(source);
  harris_detector.setNormals(source_normals);
  harris_detector.setNumberOfThreads(num_threads);
  harris_detector.setRadius(params.harris_radius_);
  harris_detector.setThreshold(params.harris_threshold_);
  harris_detector.setMethod(
      static_cast<pcl::HarrisKeypoint3D<Point, Point>::ResponseMethod>(
          params.harris_response_));
  harris_detector.compute(*source_keypoints);
}

void ComputeFeatures(const PointCloudConstPtr& keypoints,
                                                const PointCloudConstPtr& input,
                                                const Normals::Ptr& normals,
                                                const double& search_radius,
                                                const int& num_threads,
                                                Features::Ptr features) {
  pcl::search::KdTree<Point>::Ptr search_method(new pcl::search::KdTree<Point>);
  pcl::FPFHEstimationOMP<Point, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
  fpfh_est.setInputCloud(keypoints);
  fpfh_est.setSearchSurface(input);
  fpfh_est.setInputNormals(normals);
  fpfh_est.setSearchMethod(search_method);
  fpfh_est.setRadiusSearch(search_radius);
  fpfh_est.setNumberOfThreads(num_threads);
  fpfh_est.compute(*features);
}