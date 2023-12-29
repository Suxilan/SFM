# SFM
基于C++OpenCV的增量式三维重建算法（覆盖计算机视觉、数字摄影测量课程设计内容）

# Requirements
* OpenCV 4.5.5
* ceres 2.0.0
* pcl 1.13.0
### Description
> * **OpenCV：** 提供计算机视觉常用算法，包括影像读取、特征点提取与匹配、本质矩阵计算与分解、密集匹配等功能;  
> * **Ceres：** 用于非线性优化问题，类似于光束法平差中求解误差方程的最小二乘解。（*PS:如果待求解的连接点或影像外参偏多，请使用稀疏矩阵模块`suitesparse`求解以减少内存消耗*）;  
> * **pcl：** 用于点云后处理，平滑、下采样以及局部重采样使用，减少点云细碎度

emsp;emsp;还可以使用**happly**库将点云存储为ply格式，以适配更多渲染器  

# Data
emsp;emsp;将数据存放至`data`文件夹下即可。项目中提供了可供参考测试的模型数据（已经过背景处理），`cams_1`文件夹中提供了经过检校的相机内参，直接拷贝到代码中即可。
## initialization

### intrinsic

    double fx = 5487.7;
    double fy = 5487.7;
    double x0 = 2870.8, y0 = 1949.2;
    Mat intrinsic = (Mat_<double>(3, 3) << fx, 0, x0,                // 初始化所有相片的内参
                                           0, fy, y0,
                                           0, 0, 1);
### extrinsic

     Mat origin_matrix = (Mat_<double>(4, 4) <<                      // 第一张为原点
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1);

# Path Planning&Procedure Recovery
emsp;emsp;SFM最重要的问题就是航迹恢复，判断拍摄的过程，较为成熟的方法是先暴力匹配所有图片，再依据两两之间匹配点的数目建立最优化问题，通常使用的方法是**最小生成树**。但事实上，如果相片数目少，完全可以通过肉眼进行判断，而真正作业过程相片拍摄都是按一定的运动顺序进行的，只需要根据文件名编号从前往后读取即可，除非某些竞赛或者老师专门出题坑人。

*恢复过程的算法和代码我会在另一个项目中补充，这里就先略过啦！*

# SIFT
emsp;emsp;有关SIFT的知识我会放在另一个项目介绍，以及手搓SIFT、手搓GPUSIFT、手搓并行GPUSIFT。详情请见...
# Knnmatch + RANSAC
emsp;emsp;对SIFT提取特征点的描述子进行匹配吗，先利用欧氏距离进行判断后，再通过RANSAC随机抽样剔除误匹配。

### tips
如果觉得还不保险，可以利用相关系数再进行筛查，或者利用像对所有匹配点建立立体模型后通过视差进行筛查。

### result

# Bundle Adjustment

# SGBM

# Reprocess
