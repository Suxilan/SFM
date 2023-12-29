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
    // 2. ȥ����Ⱥ��
    cout << "��ʼȥ����Ⱥ�㣺 " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_filtered(new PointCloud<PointXYZRGB>);
    StatisticalOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50); // �������
    sor.setStddevMulThresh(1); // ��ֵ
    sor.filter(*cloud_filtered);
    cout << "����ȥ�����!ʣ����������� " << cloud_filtered->size() << endl;

    // 3. �����²���
    cout << "��ʼ�����²����� " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_downsampled(new PointCloud<PointXYZRGB>);
    VoxelGrid<PointXYZRGB> vg;
    vg.setInputCloud(cloud_filtered);
    vg.setLeafSize(0.01, 0.01, 0.01); // ���ش�С
    vg.filter(*cloud_downsampled);
    cout << "�����²�������!ʣ����������� " << cloud_downsampled->size() << endl;

    // 4. ƽ���˲�
    cout << "��ʼ�����˲��� " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_smoothed(new PointCloud<PointXYZRGB>);
    MovingLeastSquares<PointXYZRGB, PointXYZRGB> mls;
    mls.setInputCloud(cloud_downsampled);
    mls.setSearchRadius(0.03); // �����뾶
    mls.setPolynomialOrder(3); // ����ʽ��Ͻ���
    mls.setUpsamplingMethod(MovingLeastSquares<PointXYZRGB, PointXYZRGB>::SAMPLE_LOCAL_PLANE); // ʹ�þֲ�ƽ�����
    mls.setUpsamplingRadius(0.03); // �����뾶
    mls.setUpsamplingStepSize(0.01); // ��������
    mls.process(*cloud_smoothed);
    cout << "�����˲�����!ʣ����������� " << cloud_smoothed->size() << endl;

    cout << "��ʼȥ����Ⱥ�㣺 " << endl;
    PointCloud<PointXYZRGB>::Ptr cloud_filtered1(new PointCloud<PointXYZRGB>);
    StatisticalOutlierRemoval<PointXYZRGB> sor2;
    sor2.setInputCloud(cloud_smoothed);
    sor2.setMeanK(50); // �������
    sor2.setStddevMulThresh(0.1); // ��ֵ
    sor2.filter(*cloud_filtered1);
    cout << "����ȥ�����!ʣ����������� " << cloud_filtered1->size() << endl;

    cloud = cloud_filtered;
}

int main1()
{
    // 1. ��ȡ��������

    string foldername = "dense_3Dpoints";
    boost::filesystem::path folderPath(foldername);

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

    // �����ļ����е��ļ��������ļ���˳���ȡ
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
        cout << "���ƶ�ȡ�ɹ�!���������� " << cloud->size() << endl;

        i++;
    }

    process(cloud);

    string outputfolder = "optimized_3Dpoints";
    // ����Ż���ĵ�������
    if (_access(outputfolder.c_str(), 0) == -1)//����ֵΪ-1����ʾ������
    {
        cout << "�����ļ���" << outputfolder << endl;;
        _mkdir(outputfolder.c_str());
    }

    PCDWriter writer;
    writer.write<pcl::PointXYZRGB>(outputfolder + "/cloud" + to_string(i) + ".pcd", *cloud, false);
    cout << "�ɹ���������ļ���" << endl;
        ////// ��׼����һ��������
        //// // 5. ������׼
        ////cout << "��ʼ��׼����" << endl;
        ////PointCloud<PointXYZRGB>::Ptr cloud_source(new PointCloud<PointXYZRGB>);
        ////PointCloud<PointXYZRGB>::Ptr cloud_target(new PointCloud<PointXYZRGB>);
        ////// ����Դ���ƺ�Ŀ�����

        ////IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
        ////icp.setInputSource(cloud);
        ////icp.setInputTarget(cloud1);
        ////PointCloud<PointXYZRGB> aligned_cloud;
        ////icp.align(aligned_cloud);

        //PCDWriter writer;
        //writer.write<pcl::PointXYZRGB>(outputfolder + "/cloud"+to_string(i) + ".pcd", *cloud, false);
        //cout << "�ɹ���������ļ���" << endl;
    

    return 0;
}
