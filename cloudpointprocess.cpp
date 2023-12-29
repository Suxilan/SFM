#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <pcl/registration/icp.h>
#include <direct.h>
#include <io.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace pcl;


void process(PointCloud<PointXYZRGB>::Ptr& cloud)
{
    // 2. 去除离群点
    cout << "开始去除离群点： " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_filtered(new PointCloud<PointXYZRGB>);
    StatisticalOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50); // 邻域点数
    sor.setStddevMulThresh(1); // 阈值
    sor.filter(*cloud_filtered);
    cout << "点云去噪结束!剩余点云数量： " << cloud_filtered->size() << endl;

    // 3. 体素下采样
    cout << "开始体素下采样： " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_downsampled(new PointCloud<PointXYZRGB>);
    VoxelGrid<PointXYZRGB> vg;
    vg.setInputCloud(cloud_filtered);
    vg.setLeafSize(0.01, 0.01, 0.01); // 体素大小
    vg.filter(*cloud_downsampled);
    cout << "点云下采样结束!剩余点云数量： " << cloud_downsampled->size() << endl;

    // 4. 平滑滤波
    cout << "开始点云滤波： " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_smoothed(new PointCloud<PointXYZRGB>);
    MovingLeastSquares<PointXYZRGB, PointXYZRGB> mls;
    mls.setInputCloud(cloud_downsampled);
    mls.setSearchRadius(0.03); // 搜索半径
    mls.setPolynomialOrder(3); // 多项式拟合阶数
    mls.setUpsamplingMethod(MovingLeastSquares<PointXYZRGB, PointXYZRGB>::SAMPLE_LOCAL_PLANE); // 使用局部平面采样
    mls.setUpsamplingRadius(0.03); // 采样半径
    mls.setUpsamplingStepSize(0.01); // 采样步长
    mls.process(*cloud_smoothed);
    cout << "点云滤波结束!剩余点云数量： " << cloud_smoothed->size() << endl;

    cout << "开始去除离群点： " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_filtered1(new PointCloud<PointXYZRGB>);
    StatisticalOutlierRemoval<PointXYZRGB> sor2;
    sor2.setInputCloud(cloud_smoothed);
    sor2.setMeanK(50); // 邻域点数
    sor2.setStddevMulThresh(0.1); // 阈值
    sor2.filter(*cloud_filtered1);
    cout << "点云去噪结束!剩余点云数量： " << cloud_filtered1->size() << endl;

    cloud = cloud_filtered;
}

int main1()
{
    // 1. 读取点云数据

    string foldername = "dense_3Dpoints";
    boost::filesystem::path folderPath(foldername);

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

    // 遍历文件夹中的文件，并按文件名顺序读取
    boost::filesystem::directory_iterator endItr;
    int i = 1;
    for (boost::filesystem::directory_iterator itr(folderPath); itr != endItr; ++itr) {
      
        if (boost::filesystem::is_regular_file(itr->path())) {
            ifstream file(itr->path().string());
            if (!file.is_open()) {
                cerr << "Failed to open file: " << itr->path().string() << endl;
                continue;
            }

            float x, y, z;
            int r, g, b;
            while (file >> x >> y >> z >> r >> g >> b) {
                pcl::PointXYZRGB point;
                point.x = x;
                point.y = y;
                point.z = z;
                point.r = r;
                point.g = g;
                point.b = b;
                cloud->push_back(point);
            }
            file.close();
        }
        cout << "点云读取成功!点云数量： " << cloud->size() << endl;

        i++;
    }

    process(cloud);

    string outputfolder = "optimized_3Dpoints";
    // 输出优化后的点云数据
    if (_access(outputfolder.c_str(), 0) == -1)//返回值为-1，表示不存在
    {
        cout << "创建文件夹" << outputfolder << endl;;
        _mkdir(outputfolder.c_str());
    }

    PCDWriter writer;
    writer.write<pcl::PointXYZRGB>(outputfolder + "/cloud" + to_string(i) + ".pcd", *cloud, false);
    cout << "成功保存点云文件！" << endl;
        ////// 配准到第一个点云下
        //// // 5. 点云配准
        ////cout << "开始配准点云" << endl;
        ////PointCloud<PointXYZRGB>::Ptr cloud_source(new PointCloud<PointXYZRGB>);
        ////PointCloud<PointXYZRGB>::Ptr cloud_target(new PointCloud<PointXYZRGB>);
        ////// 设置源点云和目标点云

        ////IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
        ////icp.setInputSource(cloud);
        ////icp.setInputTarget(cloud1);
        ////PointCloud<PointXYZRGB> aligned_cloud;
        ////icp.align(aligned_cloud);

        //PCDWriter writer;
        //writer.write<pcl::PointXYZRGB>(outputfolder + "/cloud"+to_string(i) + ".pcd", *cloud, false);
        //cout << "成功保存点云文件！" << endl;
    

    return 0;
}
