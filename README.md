# SFM
基于C++OpenCV的增量式三维重建算法（覆盖计算机视觉、数字摄影测量课程设计内容）

# requirements
* OpenCV 4.5.5
* ceres 2.0.0
* pcl 1.13.0
### Description
**OpenCV：** 提供计算机视觉常用算法，包括影像读取、特征点提取与匹配、本质矩阵计算与分解、密集匹配等功能;  
**Ceres：** 用于非线性优化问题，类似于光束法平差中求解误差方程的最小二乘解。（PS:如果待求解的连接点或影像外参偏多，请使用稀疏矩阵模块`suitesparse`求解以减少内存消耗);  
**pcl：** 用于点云后处理，平滑、下采样以及局部重采样使用，减少点云细碎度  
（还可以使用happly库将点云存储为ply格式，以适配更多渲染器）  

# Data
将数据存放至`data`文件夹下即可。项目中提供了可供参考测试的模型数据（已经过背景处理），`cams_1`文件夹中提供了经过检校的相机内参，直接拷贝到代码中即可。
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
