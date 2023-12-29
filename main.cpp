#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <direct.h>
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
#include "bundle_adjustment.h"

using namespace cv;
using namespace std;
using namespace pcl;

// 相机、相片参数
int imagenums;

/**
 * @brief 读取影像
 * @param path 影像路径
 * @return Mat类影像
*/
Mat readimg(string path)
{
    // 读取RGB图像
    Mat img = imread(path, IMREAD_COLOR);
    //检测图像是否加载成功
    if (img.empty() || img.depth() != CV_8U)
    {
        CV_Error(CV_StsBadArg, "输入图像为空，或者图像深度不是CV_8U");
        return Mat();
    }

    return img;
}

/**
 * @brief 创建航带影像
 * @param folderPath 文件夹路径
 * @param images Mat类影像
 * @param filenames 影像名称
*/
void create_imglines(
    const string folderPath,
    vector<Mat>& images,
    vector<string>& filenames)
{
    string pattern = folderPath + "/*.png";  // 假设文件夹中只有 jpg 格式的图片
    vector<String> imagePaths;
    glob(pattern, imagePaths);

    // 遍历图片文件并读取到向量中
    for (const auto& imagePath : imagePaths) {
        Mat image = readimg(imagePath);
        if (!image.empty()) {
            images.push_back(image);

            // 提取影像文件名并去除扩展名后存入向量
            size_t lastSlashIndex = imagePath.find_last_of("/\\");
            if (lastSlashIndex != string::npos) {
                String imageName = imagePath.substr(lastSlashIndex + 1);
                size_t dotIndex = imageName.find_last_of(".");
                if (dotIndex != string::npos) {
                    imageName = imageName.substr(0, dotIndex);
                }
                filenames.push_back(imageName);
            }
        }     
    }
    imagenums = images.size();
    cout << endl << "航片数量：" << imagenums << endl;
}

/**
 * @brief 初始化相机参数
 * @param exter_matrix 相机外参
 * @param inner_matrix 相机内参
*/
void init_params(
    vector<Mat>& exter_matrix,
    vector<Mat>& inner_matrix)
{
    // 此处可以写文本读取
    // ...
    // 初始化
    Mat origin_matrix = (Mat_<double>(4, 4) <<                      // 第一张为原点
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);

    double fx = 2892.33;
    double fy = 2883.18;
    double x0 = 823.205, y0 = 619.071;
    //double fx = 5487.7;
    //double fy = 5487.7;
    //double x0 = 2870.8, y0 = 1949.2;
    Mat intrinsic = (Mat_<double>(3, 3) << fx, 0, x0,                // 初始化所有相片的内参
        0, fy, y0,
        0, 0, 1);

    for (int i = 0; i < imagenums; i++)
    {
        inner_matrix.push_back(intrinsic);
    }
    exter_matrix.resize(imagenums);
    exter_matrix[0] = origin_matrix;
}

/**
 * @brief 计算特征点
 * @param imgs 影像
 * @param filenames 影像名称
 * @param kpts 特征点
 * @param descriptors 特征点描述子
*/
void cal_keypoints(
    const vector<Mat> imgs,
    const vector<string> filenames,
    vector<vector<KeyPoint>> & kpts,
    vector<Mat> & descriptors)
{
    for (int i = 0; i < imagenums; i++)
    {
        cout << "正在计算第 " << i + 1 << " 张影像: " << filenames[i];
        Mat img = imgs[i];
        vector<KeyPoint> kpt;
        Mat descriptor;
        // --------------------------------
        // sift特征检测,提取1000个特征点
        Ptr<SIFT> detector = SIFT::create();
        detector->detectAndCompute(img, Mat(), kpt, descriptor);
        kpts.push_back(kpt);
        descriptors.push_back(descriptor);
        cout << "点数: " << kpt.size() << endl;
    }
}


/**
 * @brief 进行影响间两两匹配
 * @param kpts 特征点
 * @param descriptors 特征点描述子
 * @param inner_matrix 相机内参
 * @param filenames 影像名称
 * @param matches 匹配的特征点索引
*/
void match(
    const vector<Mat> img,
    const vector<vector<KeyPoint>> keypoints,
    const vector<Mat> descriptors,
    const vector<Mat> inner_matrix,
    const vector<string> filenames,
    vector<vector<DMatch>>& allmatches)
{
    for (int i = 0; i < imagenums ; i++)
    {
        int j = i + 1;
        if (j == imagenums)
        {
            j = 0;
        }
        cout << "正在匹配像对: " << filenames[i] << "-" << filenames[j] << endl;
        Mat descriptorsL = descriptors[i];
        Mat descriptorsR = descriptors[j];
        vector<KeyPoint> keypointsL = keypoints[i];
        vector<KeyPoint> keypointsR = keypoints[j];
        Mat cameramatrix = inner_matrix[i];
        // =============================================================
        Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
        // 匹配描述符
        vector<vector<DMatch>> matches;
        matcher->knnMatch(descriptorsL, descriptorsR, matches, 2);

        // 距离筛选
        vector<DMatch> goodMatches;
        vector<Point2d> matchpointsL, matchpointsR;
        double ratioThreshold = 0.7; // 设定一个阈值，用于筛选匹配对
        for (size_t k = 0; k < matches.size(); ++k)
        {
            if (matches[k][0].distance < ratioThreshold * matches[k][1].distance)
            {
                matchpointsL.push_back(keypointsL[matches[k][0].queryIdx].pt);
                matchpointsR.push_back(keypointsR[matches[k][0].trainIdx].pt);
                goodMatches.push_back(matches[k][0]);
            }
        }

        //FlannBasedMatcher  matcher;
        //std::vector<cv::DMatch> matches;
        //matcher.match(descriptorsL, descriptorsR, matches);

        //// 可选：筛选最佳匹配
        //std::sort(matches.begin(), matches.end());
        //std::vector<cv::DMatch> goodMatches;
        //vector<Point2d> matchpointsL, matchpointsR;
        //float distanceThreshold = 100.0;
        //for (size_t k = 0; k < matches.size(); ++k) {
 
        //    if (matches[i].distance < distanceThreshold) {
        //        matchpointsL.push_back(keypointsL[matches[k].queryIdx].pt);
        //        matchpointsR.push_back(keypointsR[matches[k].trainIdx].pt);
        //        goodMatches.push_back(matches[k]);
        //    }
        //}


        cout << "匹配个数:" << goodMatches.size() << "对" << endl;

        // RANSAC剔除粗差
        vector<DMatch> selectMatches;
        vector<uchar> inliers(matchpointsL.size(), 0);
        Mat EssentialMat = findEssentialMat(matchpointsL, matchpointsR, cameramatrix, FM_RANSAC, 0.99, 1.0, inliers);
        for (size_t k = 0; k < goodMatches.size(); k++) {
            if (inliers[k] != 0)
            {
                selectMatches.push_back(goodMatches[k]);
            }
        }
        cout << "ransac匹配成功：" << selectMatches.size() << "对" << endl;
 
        //// 相关系数筛选
        //vector<DMatch> correlationFilteredMatches;
        //double correlationThreshold = 0.9; // 相关系数的阈值
        //Size windowSize(21, 21); // 窗口大小为21x21
        //int border = windowSize.width / 2; // 边界大小，用于判断窗口是否越界
        //for (const DMatch& match : selectMatches)
        //{
        //    Point2f ptL = keypointsL[match.queryIdx].pt;
        //    Point2f ptR = keypointsR[match.trainIdx].pt;

        //    // 边界检查
        //    if (ptL.x < border || ptL.x >= img[i].cols - border ||
        //        ptL.y < border || ptL.y >= img[i].rows - border ||
        //        ptR.x < border || ptR.x >= img[j].cols - border ||
        //        ptR.y < border || ptR.y >= img[j].rows - border)
        //    {
        //        continue; // 跳过越界的点
        //    }

        //    // 计算窗口内的相关系数
        //    Mat windowL(img[i], Rect(ptL.x - windowSize.width / 2, ptL.y - windowSize.height / 2, windowSize.width, windowSize.height));
        //    Mat windowR(img[j], Rect(ptR.x - windowSize.width / 2, ptR.y - windowSize.height / 2, windowSize.width, windowSize.height));
        //    Mat correlationResult;
        //    matchTemplate(windowL, windowR, correlationResult, TM_CCOEFF_NORMED);
        //    double correlation = correlationResult.at<float>(0, 0);


        //    if (correlation >= correlationThreshold)
        //    {
        //        correlationFilteredMatches.push_back(match);
        //    }
        //}
        //cout << "相关系数筛选后匹配个数：" << correlationFilteredMatches.size() << "对" << endl;

        allmatches.push_back(selectMatches);
    }
}

/**
 * @brief 绘制拼接影像
 * @param imageL 左像
 * @param imageR 右像
 * @param cornersL 左像角点
 * @param cornersR 右像角点
 * @return 拼接影像
*/
Mat drawmatches(
    Mat imageL, 
    Mat imageR, 
    vector<KeyPoint>cornersL, 
    vector<KeyPoint>cornersR)
{
    // 绘制角点
    Mat CornerL_Detection;
    drawKeypoints(imageL, cornersL, CornerL_Detection, Scalar(0, 255, 255));
    Mat CornerR_Detection;
    drawKeypoints(imageR, cornersR, CornerR_Detection, Scalar(0, 255, 255));

    //将影像拼接在一起
    // 计算拼接后的影像尺寸
    int max_width = CornerL_Detection.cols + CornerR_Detection.cols;
    int max_height = max(CornerL_Detection.rows, CornerR_Detection.rows);

    Mat mergeline(max_height, max_width, CornerL_Detection.type());
    mergeline.setTo(Scalar::all(0)); // 用0填充
    // 复制影像到拼接后的影像中
    Mat roi1(mergeline, Rect(0, 0, CornerL_Detection.cols, CornerL_Detection.rows));
    CornerL_Detection.copyTo(roi1);
    Mat roi2(mergeline, Rect(CornerL_Detection.cols, 0, CornerR_Detection.cols, CornerR_Detection.rows));
    CornerR_Detection.copyTo(roi2);

    // 检查图像通道数
    if (mergeline.channels() == 1)
    {
        // 图像为灰度图像，进行通道变换
        cvtColor(mergeline, mergeline, COLOR_GRAY2BGR);
    }
    //生成随机数
    // 获取当前时间
    time_t now = time(0);
    // 用当前时间作为种子来初始化随机数生成器
    RNG rng(now);
    for (int i = 0; i < cornersL.size(); i++)
    {
        line(mergeline, Point(cornersL[i].pt.x, cornersL[i].pt.y), Point(cornersR[i].pt.x + imageL.cols, cornersR[i].pt.y), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, 0);
    }
    // 显示结果
    return mergeline;
}

/**
 * @brief 输出匹配影像
 * @param imgs 原图
 * @param allmatches 匹配索引
 * @param keypoints 特征点
 * @param foldername 匹配图文件夹
 * @param filenames 文件名
 * @param matchimg 匹配图
*/
void writematchimg(
    const vector<Mat> imgs, 
    const vector<vector<DMatch>>& allmatches, 
    const vector<vector<KeyPoint>> keypoints, 
    const string foldername,
    const vector<string> filenames,
    vector<Mat>& matchimg)
{
    for (int i = 0; i < imagenums; i++)
    {
        int j = i + 1;
        if (j == imagenums)
        {
            j = 0;
        }

        Mat imgL, imgR;
        vector<KeyPoint> kpL, kpR, keypointsL, keypointsR;
        vector<DMatch> matches;
        imgL = imgs[i]; imgR = imgs[j];
        matches = allmatches[i];
        keypointsL = keypoints[i];
        keypointsR = keypoints[j];

        kpL.resize(matches.size());
        kpR.resize(matches.size());
        for (size_t k = 0; k < matches.size(); ++k)
        {
            // 获取匹配对的索引和距离
            kpL[k] = keypointsL[matches[k].queryIdx];
            kpR[k] = keypointsR[matches[k].trainIdx];
            matches[k].queryIdx = k;
            matches[k].trainIdx = k;
        }
        Mat matchedImage;
        string matchids = filenames[i] + "_" + filenames[j];
        matchedImage = drawmatches(imgL, imgR, kpL, kpR);
        //namedWindow("Matched Image"+ matchids, 2);
        //imshow("Matched Image" + matchids, matchedImage);
        if (_access(foldername.c_str(), 0) == -1)            
            //返回值为-1，表示不存在
        {
            _mkdir(foldername.c_str());
        }
        imwrite(foldername+"/Matches" + matchids + ".jpg", matchedImage);
        //waitKey(0);
    }
}

/**
 * @brief 光束法区域网平差
 * @param keypoints 每张相片的特征点
 * @param matches 特征点匹配索引
 * @param exter_matrix 相机外参
 * @param inner_matrix 相机内参
 * @param pts_3d 对应物方点坐标
*/
void bundle_adjustment(
    const vector<vector<KeyPoint>> keypoints,
    const vector<vector<DMatch>> matches,
    vector<Mat>& exter_matrix,
    vector<Mat>& inner_matrix,
    vector<vector<Point3d>>& pts_3d)
{
    Mat Rl = exter_matrix[0](Range(0, 3), Range(0, 3)).clone();
    Mat Tl = exter_matrix[0](Range(0, 3), Range(3, 4)).clone();

    // 前两张先相对定向求RT
    vector<KeyPoint>kp1s, kp2s;
    kp1s.resize(matches[0].size());
    kp2s.resize(matches[0].size());
    for (size_t j = 0; j < kp1s.size(); ++j)
    {
        // 获取匹配对的索引和距离
        kp1s[j] = keypoints[0][matches[0][j].queryIdx];
        kp2s[j] = keypoints[1][matches[0][j].trainIdx];
    }

    // 相对定向
    vector<Point2d> pointsL, pointsR;
    for (size_t i = 0; i < kp1s.size(); i++)
    {
        pointsL.push_back(kp1s[i].pt);
        pointsR.push_back(kp2s[i].pt);
    }

    Mat EssentialMat = findEssentialMat(pointsL, pointsR, inner_matrix[0], RANSAC);
    Mat R0, T0;
    recoverPose(EssentialMat, pointsL, pointsR, inner_matrix[0], R0, T0);

    Mat Rr, Tr;
    Rr = Rl * R0;
    Tr = Tl + Rl * T0;                                          // 定向到第一张影像坐标系下
    // 得到初始RT后正式开始前方交会后方交会流程
    // 总结为:
    // 1.得到RT;
    // 2.前方交会求3D点;
    // 3.光束法重投影
    // 4.后方交会

    for (size_t i = 0; i < imagenums - 1; ++i)
    {
        int j = i + 2;
        if (j == imagenums)
        {
            j = 0;
        }
        cout << "计算像对" << i + 1 << endl;
        vector<KeyPoint>kps1, kps2, kps3;
        // 前方交会点的要求是,前两张的点在第三张上出现
        for (size_t m = 0; m < matches[i].size(); ++m)
        {
            KeyPoint kp1, kp2, kp3;
            for (size_t n = 0; n < matches[i + 1].size(); ++n)
            {
                if (matches[i][m].trainIdx == matches[i + 1][n].queryIdx)
                {
                    kp1 = keypoints[i][matches[i][m].queryIdx];
                    kp2 = keypoints[i + 1][matches[i][m].trainIdx];
                    kp3 = keypoints[j][matches[i + 1][n].trainIdx];
                    kps1.push_back(kp1); kps2.push_back(kp2); kps3.push_back(kp3);
                    continue;
                }
            }
        }
        cout << "参加光束法的点数：" << kps1.size() << endl;
        vector<Point3d>pt_3d;                                        // 前方交会得3D点坐标
        vector<vector<double>> point3d;
        Triangulation(inner_matrix[i], inner_matrix[i + 1], kps1, kps2, Rl, Tl, Rr, Tr, pt_3d);
        cout << Rr << endl;
        cout << Tr << endl;
        for (size_t j = 0; j < pt_3d.size(); ++j)
        {
            vector<double> point;
            point.resize(3);
            point[0] = pt_3d[j].x;
            point[1] = pt_3d[j].y;
            point[2] = pt_3d[j].z;
            point3d.push_back(point);
        }

        vector<double> camera_params;
        vector<double> camera_matrix;
        Vec3d rvec;
        Rodrigues(Rr, rvec);                                         // 将旋转矩阵 R 转换为欧拉角旋转向量
        // 读取外参
        camera_params.insert(camera_params.end(), rvec.val, rvec.val + 3);
        camera_params.insert(camera_params.end(), Tr.begin<double>(), Tr.end<double>());
        // 读取内参
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(0, 0));
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(1, 1));
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(0, 2));
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(1, 2));
        Bundle_adjustment(kps1, kps2, camera_params, camera_matrix, point3d);

        //得到平差后的外参矩阵
        copy(camera_params.begin(), camera_params.begin() + 3, rvec.val);
        copy(camera_params.begin() + 3, camera_params.end(), Tr.ptr<double>());
        Rodrigues(rvec, Rr);
        Mat extrinsic = Mat::eye(4, 4, CV_64F);                     // 创建外参矩阵 extrinsic
        Mat R_roi = extrinsic(Rect(0, 0, 3, 3));
        Rr.copyTo(R_roi);
        Mat T_roi = extrinsic(Rect(3, 0, 1, 3));
        Tr.copyTo(T_roi);
        exter_matrix[i + 1] = extrinsic;      
        Rl = Rr; Tl = Tr;                                           // 外参改为下一张影像外参
        //得到平差后的内参矩阵
        Mat intrinsic = Mat::eye(3, 3, CV_64F);                     // 创建外参矩阵 intrinsic
        intrinsic.at<double>(0, 0) = camera_matrix[0];
        intrinsic.at<double>(1, 1) = camera_matrix[1];
        intrinsic.at<double>(0, 2) = camera_matrix[2];
        intrinsic.at<double>(1, 2) = camera_matrix[3];
        inner_matrix[i + 1] = intrinsic;

        //得到平差后的3d坐标
        for (size_t j = 0; j < point3d.size(); ++j)
        {
            pt_3d[j].x = point3d[j][0];
            pt_3d[j].y = point3d[j][1];
            pt_3d[j].z = point3d[j][2];
        }
        pts_3d.push_back(pt_3d);
        
        //开始后方交会

        Mat rvec1;  // 旋转向量
        Mat tvec1;  // 平移向量
  
        Resection(inner_matrix[j], kps3, pt_3d, rvec1, tvec1);
        Mat R; // 旋转矩阵
        Rodrigues(rvec1, R);
        Rr = R; Tr = tvec1;
 
        // 回环
        if (i == imagenums - 2)
        {
            cout << "计算像对" << i + 2 << endl;
            vector<KeyPoint>kps1, kps2;
            for (size_t k = 0; k < matches[i + 1].size(); ++k)
            {
                KeyPoint kp1, kp2;                      // 最后两张
                kp1 = keypoints[i + 1][matches[i + 1][k].queryIdx];
                kp2 = keypoints[j][matches[i + 1][k].trainIdx];
                kps1.push_back(kp1); kps2.push_back(kp2);
            }
  
            vector<Point3d>pt_3d;                                    // 前方交会得3D点坐标
            vector<vector<double>> point3d;
            Triangulation(inner_matrix[i + 1], inner_matrix[j], kps1, kps2, Rl, Tl, Rr, Tr, pt_3d);
            for (size_t j = 0; j < pt_3d.size(); ++j)
            {
                vector<double> point;
                point.resize(3);
                point[0] = pt_3d[j].x;
                point[1] = pt_3d[j].y;
                point[2] = pt_3d[j].z;
                point3d.push_back(point);
            }

            vector<double> camera_params;
            vector<double> camera_matrix;
            Vec3d rvec;
            Rodrigues(Rr, rvec);                                         // 将旋转矩阵 R 转换为欧拉角旋转向量
            // 读取外参
            camera_params.insert(camera_params.end(), rvec.val, rvec.val + 3);
            camera_params.insert(camera_params.end(), Tr.begin<double>(), Tr.end<double>());
            // 读取内参
            camera_matrix.push_back(inner_matrix[j].at<double>(0, 0));
            camera_matrix.push_back(inner_matrix[j].at<double>(1, 1));
            camera_matrix.push_back(inner_matrix[j].at<double>(0, 2));
            camera_matrix.push_back(inner_matrix[j].at<double>(1, 2));
            Bundle_adjustment(kps1, kps2, camera_params, camera_matrix, point3d);

            //得到平差后的外参矩阵
            copy(camera_params.begin(), camera_params.begin() + 3, rvec.val);
            copy(camera_params.begin() + 3, camera_params.end(), Tr.ptr<double>());
            Rodrigues(rvec, Rr);
            Mat extrinsic = Mat::eye(4, 4, CV_64F);                     // 创建外参矩阵 extrinsic
            Mat R_roi = extrinsic(Rect(0, 0, 3, 3));
            Rr.copyTo(R_roi);
            Mat T_roi = extrinsic(Rect(3, 0, 1, 3));
            Tr.copyTo(T_roi);
            exter_matrix[j] = extrinsic;

            //得到平差后的内参矩阵
            Mat intrinsic = Mat::eye(3, 3, CV_64F);                     // 创建外参矩阵 intrinsic
            intrinsic.at<double>(0, 0) = camera_matrix[0];
            intrinsic.at<double>(1, 1) = camera_matrix[1];
            intrinsic.at<double>(0, 2) = camera_matrix[2];
            intrinsic.at<double>(1, 2) = camera_matrix[3];
            inner_matrix[j] = intrinsic;

            //得到平差后的3d坐标
            for (size_t j = 0; j < point3d.size(); ++j)
            {
                pt_3d[j].x = point3d[j][0];
                pt_3d[j].y = point3d[j][1];
                pt_3d[j].z = point3d[j][2];
            }
            pts_3d.push_back(pt_3d);
        }
    }
}

/**
 * @brief 写入相机参数
 * @param camera_intrinsics 相机内参
 * @param camera_extrinsics 相机外参
 * @param foldername 文件夹名称
 * @param filenames 文件名
*/
void writeCameraParams(
    const vector<Mat>& camera_intrinsics,
    const vector<Mat>& camera_extrinsics,
    const string foldername,
    const vector<string> filenames) 
{

    if (_access(foldername.c_str(), 0) == -1)
        //返回值为-1，表示不存在
    {
        _mkdir(foldername.c_str());
    }

    // 设置输出的字段宽度和精度
    const int field_width = 15;
    const int precision = 6;

    // 循环遍历每个相机的内参和外参
    for (size_t i = 0; i < imagenums; ++i) {
        string output_file = foldername + "/" + filenames[i] + "_params.txt";

        ofstream outfile(output_file);

        if (!outfile.is_open()) {
            std::cerr << "Failed to open the output file." << endl;
            return;
        }
        const Mat& intrinsic = camera_intrinsics[i];
        const Mat& extrinsic = camera_extrinsics[i];

        outfile << "image-" << i + 1 << " intrinsic parameters:" << endl;
        outfile << fixed << setprecision(precision) << intrinsic << endl;

        outfile << "image-" << i + 1 << " extrinsic parameters:" << endl;
        outfile << fixed << setprecision(precision) << extrinsic << endl;

        outfile << endl;
        outfile.close();
    }
    cout << "写入参数成功！" << endl;
}

/**
 * @brief 写入稀疏物方点坐标
 * @param pts_3d 物方点
 * @param foldername 文件夹
 * @param filenames 文件名
*/
void write_3dpts(
    const vector<vector<Point3d>> pts_3d,
    const string foldername,
    const vector<string> filenames)
{
    if (_access(foldername.c_str(), 0) == -1)
        //返回值为-1，表示不存在
    {
        _mkdir(foldername.c_str());
    }

    // 设置输出的字段宽度和精度
    const int field_width = 15;
    const int precision = 6;
    for (size_t i = 0; i < imagenums; ++i)
    {
        int k = i + 1;
        if (k == imagenums)
        {
            k = 0;
        }
        string output_file = foldername + "/" + filenames[i] + "_"+ filenames[k]+ "_sparse3d.txt";

        ofstream outfile(output_file);

        if (!outfile.is_open()) {
            cerr << "Failed to open the output file." << endl;
            return;
        }
        for (size_t j = 0; j < pts_3d[i].size(); ++j)
        {
            outfile << fixed << setprecision(precision) <<
                pts_3d[i][j].x << "\t" << pts_3d[i][j].y << "\t" << pts_3d[i][j].z << "\t" << endl;
        }
    }
    cout << "稀疏物方点保存成功！" << endl;
}

/**
 * @brief 创建核线像对
 * @param imgs 原始图像
 * @param keypoints 每张图特征点
 * @param matches 特征点匹配索引
 * @param extrinsics 外参
 * @param intrinsics 内参
 * @param eplimgls 左核线影像
 * @param eplimgrs 右核线影像
 * @param Qs 深度矩阵
 * @param img2epls 左像变换旋转矩阵
 * @param img2eprs 右像变换旋转矩阵
 * @param epikpls 左核线影像特征点
 * @param epikprs 右核线影像特征点
*/
void create_epimg(
    const vector<Mat> imgs,
    const vector<vector<KeyPoint>> keypoints,
    const vector<vector<DMatch>> matches,
    const vector<Mat> extrinsics,
    const vector<Mat> intrinsics,
    vector<Mat>& eplimgls, 
    vector<Mat>& eplimgrs,
    vector<Mat>& Qs,
    vector<Mat>& img2epls, 
    vector<Mat>& img2eprs,
    vector<vector<KeyPoint>>& epikpls,
    vector<vector<KeyPoint>>& epikprs)
{
    for (int i = 0; i < imagenums; i++)
    {
        int j = i + 1;
        if (j == imagenums)
        {
            j = 0;
        }

        Mat imgl = imgs[i]; Mat imgr = imgs[j];

        //获取每张图像的绝对R,T
        Mat RTL, RTR;                                         //两张图片外参
        Mat RT;                                               //两张图象相对的RT
        RTL = extrinsics[i]; RTR = extrinsics[j];
        RT = RTR * (RTL.inv());

        Mat rvecsMat, tvecsMat;                               //相对的旋转矩阵和平移矩阵
        rvecsMat = RT(Range(0, 3), Range(0, 3)).clone();
        tvecsMat = RT(Range(0, 3), Range(3, 4)).clone();

        // 内参
        Mat intrinsicl = intrinsics[i];
        Mat intrinsicr = intrinsics[j];
        Mat distCoeffsL = Mat::zeros(1, 5, CV_64FC1);
        Mat distCoeffsR = Mat::zeros(1, 5, CV_64FC1);         // 摄像机的5个畸变系数：k1,k2,p1,p2,k3
        
        Size image_size = imgl.size();                        // 图像尺寸
        Mat dstL, dstR;
        Mat RvecsL(3, 3, CV_64FC1), RvecsR(3, 3, CV_64FC1);   // 立体校正后的旋转矩阵
        Mat PvecsL(3, 4, CV_64FC1), PvecsR(3, 4, CV_64FC1);   // 立体校正后的映射矩阵
        Mat Qvecs(4, 4, CV_64FC1);                            // 立体校正后的深度矩阵
        stereoRectify(intrinsicl, distCoeffsL,
            intrinsicr, distCoeffsR,
            image_size, rvecsMat, tvecsMat,
            RvecsL, RvecsR,
            PvecsL, PvecsR, Qvecs, CALIB_ZERO_DISPARITY, -1);

        Qs.push_back(Qvecs);
        img2epls.push_back(RvecsL); 
        img2eprs.push_back(RvecsR);

        Mat rmapLfirst, rmapLsec;                             //左片的两个映射变换
        Mat rmapRfirst, rmapRsec;                             //右片的两个映射变换
        Mat imgEPL = Mat(imgl.rows, imgl.cols, CV_8UC3);      //变换后的左右图像
        Mat imgEPR = Mat(imgr.rows, imgr.cols, CV_8UC3);      //变换后的左右图像

        initUndistortRectifyMap(intrinsicl, distCoeffsL, RvecsL, PvecsL, image_size, CV_16SC2, rmapLfirst, rmapLsec);
        initUndistortRectifyMap(intrinsicr, distCoeffsR, RvecsR, PvecsR, image_size, CV_16SC2, rmapRfirst, rmapRsec);
        remap(imgl, imgEPL, rmapLfirst, rmapLsec, INTER_AREA);
        remap(imgr, imgEPR, rmapRfirst, rmapRsec, INTER_AREA);

        eplimgls.push_back(imgEPL);
        eplimgrs.push_back(imgEPR);

        vector<KeyPoint>leftKeypoints, rightKeypoints;
        for (size_t k = 0; k < matches[i].size(); ++k)
        {
            KeyPoint kp1, kp2;         
            kp1 = keypoints[i][matches[i][k].queryIdx];
            kp2 = keypoints[j][matches[i][k].trainIdx];
            leftKeypoints.push_back(kp1); rightKeypoints.push_back(kp2);
        }

        vector<KeyPoint> kpL_corrected, kpR_corrected;                  // 定义校正后的关键点集合
        kpL_corrected.resize(leftKeypoints.size());
        kpR_corrected.resize(rightKeypoints.size());
        // 定义原始图像上的关键点坐标矩阵
        Mat kpL_coords(leftKeypoints.size(), 1, CV_32FC2);
        Mat kpR_coords(rightKeypoints.size(), 1, CV_32FC2);

        // 将原始图像上的关键点坐标填充到矩阵中
        for (int i = 0; i < leftKeypoints.size(); ++i) {
            kpL_coords.at<Vec2f>(i, 0) = Vec2f(leftKeypoints[i].pt.x, leftKeypoints[i].pt.y);
            kpR_coords.at<Vec2f>(i, 0) = Vec2f(rightKeypoints[i].pt.x, rightKeypoints[i].pt.y);
        }

        // 定义校正后的关键点坐标矩阵
        Mat kpL_corrected_coords, kpR_corrected_coords;

        // 使用矩阵运算进行校正
        undistortPoints(kpL_coords, kpL_corrected_coords, intrinsicl, distCoeffsL, RvecsL, PvecsL);
        undistortPoints(kpR_coords, kpR_corrected_coords, intrinsicr, distCoeffsR, RvecsR, PvecsR);

        // 将校正后的关键点坐标转换为关键点对象
        for (int k = 0; k < kpL_corrected_coords.rows; ++k) {
            KeyPoint ckpl, ckpr;
            ckpl.pt.x = kpL_corrected_coords.at<Vec2f>(k, 0)[0];
            ckpl.pt.y = kpL_corrected_coords.at<Vec2f>(k, 0)[1];
            ckpr.pt.x = kpR_corrected_coords.at<Vec2f>(k, 0)[0];
            ckpr.pt.y = kpR_corrected_coords.at<Vec2f>(k, 0)[1];

            if (ckpl.pt.x >= 0 && ckpl.pt.x < eplimgls[i].cols && 
                ckpl.pt.y >= 0 && ckpl.pt.y < eplimgls[i].rows && 
                ckpr.pt.x >= 0 && ckpr.pt.x < eplimgrs[i].cols &&
                ckpr.pt.y >= 0 && ckpr.pt.y < eplimgrs[i].rows) {
                kpL_corrected[k] = ckpl;
                kpR_corrected[k] = ckpr;            
            }
            else {
                // Skip points outside the image boundaries
                continue;
            }
        }
        epikpls.push_back(kpL_corrected);
        epikprs.push_back(kpR_corrected);
    } 
}

/**
 * @brief 计算核线影像视差
 * @param leftKeypoints 左像特征点
 * @param rightKeypoints 右像特征点
 * @param mindisps 最小视差
 * @param maxdisps 最大视差
*/
void cal_disp(
    const vector<vector<KeyPoint>>epikpls,
    const vector<vector<KeyPoint>>epikprs,
    vector<int>& mindisps,
    vector<int>& maxdisps)
{
    for (int i = 0; i < imagenums; i++)
    {
        int j = i + 1;
        if (j == imagenums)
        {
            j = 0;
        }

        vector<KeyPoint> leftKeypoints = epikpls[i];
        vector<KeyPoint> rightKeypoints = epikprs[i];
        // 计算最小、最大视差值
        double maxDisparity = -100000.0;
        double minDisparity = 100000.0;
        vector<KeyPoint> kpL, kpR;
        for (size_t k = 0; k < leftKeypoints.size(); ++k)
        {
            // 计算视差（X轴方向上的位移）
            double disparity = leftKeypoints[k].pt.x - rightKeypoints[k].pt.x;

            // 更新最大视差值
            if (disparity > maxDisparity)
                maxDisparity = disparity;
            if (disparity <= minDisparity)
                minDisparity = disparity;
        }
        mindisps.push_back((int)minDisparity);
        maxdisps.push_back((int)maxDisparity + 1);
    }
}


/**
 * @brief 写入核线影像
 * @param eplimgls 左核线影像
 * @param eplimgrs 右核线影像
 * @param epikpls 左核线影像特征点
 * @param epikprs 右核线影像特征点
 * @param foldername 文件夹
 * @param filenames 文件名
*/
void write_epipolar_image(
    const vector<Mat> eplimgls,
    const vector<Mat> eplimgrs,
    const vector<vector<KeyPoint>>& epikpls,
    const vector<vector<KeyPoint>>& epikprs,
    const string foldername,
    const vector<string> filenames)
{
    for (int i = 0; i < imagenums; i++)
    {
        int j = i + 1;
        if (j == imagenums)
        {
            j = 0;
        }
        // 创建拼接后的图像
        Mat stitchedImage = drawmatches(eplimgls[i], eplimgrs[i], epikpls[i], epikprs[i]);

        if (_access(foldername.c_str(), 0) == -1)//返回值为-1，表示不存在
        {
            cout << "创建文件夹" << foldername << endl;;
            _mkdir(foldername.c_str());
        }

        imwrite(foldername + "/epipolar_image_" + filenames[i]+ "-" + filenames[j] + ".jpg", stitchedImage);
    }
    cout << "核线影像保存成功！" << endl;
}

/**
 * @brief 密集匹配
 * @param eplimgls 左核线影像
 * @param eplimgrs 右核线影像
 * @param mindisps 最小视差
 * @param maxdisps 最大视差
 * @param filenames 文件名
 * @param disparityMaps 视差图
*/
void dense_match(
    const vector<Mat> eplimgls,
    const vector<Mat> eplimgrs,
    const vector<int>& mindisps,
    const vector<int>& maxdisps,
    const vector<string>& filenames,
    vector<Mat>& disparityMaps)
{
    for (int i = 0; i < imagenums; i++)
    {
        int j = i + 1;
        if (j == imagenums)
        {
            j = 0;
        }
        int minDisparity = mindisps[i];
        int maxDisparity = maxdisps[i];
        cout << "最小视差为：" << minDisparity << endl;
        cout << "最大视差为：" << maxDisparity << endl;

        Mat leftImage = eplimgls[i];
        Mat rightImage = eplimgrs[i];
        Mat grayL, grayR;
        cvtColor(leftImage, grayL, COLOR_RGB2GRAY);
        cvtColor(rightImage, grayR, COLOR_RGB2GRAY);
        Size imgSize = grayL.size();
        //if ((minDisparity + maxDisparity) > 0)
        //{
        //    int minDisparity = (mindisps[i] < 30) ? mindisps[i] : ((mindisps[i] > 60) ? (mindisps[i] / 2) : mindisps[i]);
        //    if (minDisparity < 0) minDisparity = 0;
        //}
       
 
        int nmDisparities = (abs(maxDisparity - minDisparity) + 15) & -16;  //视差搜索范围
        int pngChannels = grayL.channels();                                 //获取左视图通道数
        int winSize = 3;

        Ptr<StereoSGBM> sgbm = cv::StereoSGBM::create();
        // 定义SGBM参数
        sgbm->setPreFilterCap(31);                                          //预处理滤波器截断值
        sgbm->setBlockSize(winSize);                                        //SAD窗口大小
        sgbm->setP1(8 * pngChannels * winSize * winSize);                   //控制视差平滑度第一参数
        sgbm->setP2(32 * pngChannels * winSize * winSize);                  //控制视差平滑度第二参数
        sgbm->setMinDisparity(minDisparity);                                //最小视差
        sgbm->setNumDisparities(nmDisparities);                             //视差搜索范围
        sgbm->setUniquenessRatio(10);                                       //视差唯一性百分比
        sgbm->setSpeckleWindowSize(64);                                     //散斑窗口大小
        sgbm->setSpeckleRange(2);                                           //散斑范围，会隐式乘以 16
        sgbm->setDisp12MaxDiff(1);                                          //左右视差图最大容许差异,左右一致性
        sgbm->setMode(StereoSGBM::MODE_HH);                                 //采用全尺寸双通道动态编程算法
        Mat disparityMap;
        sgbm->compute(grayL, grayR, disparityMap);

        Mat disp32F = Mat(disparityMap.rows, disparityMap.cols, CV_32FC1);
        disparityMap.convertTo(disp32F, CV_32FC1, 1.0 / 16);                //除以16得到真实视差
        // 视差图处理
        Mat filteredDisparity1 = Mat::zeros(disp32F.size(), disp32F.type());
        Mat filteredDisparity2 = Mat::zeros(disp32F.size(), disp32F.type());

        // 双边滤波有压制低频信息的特点，对视差图为负的要取反后处理
        if ((minDisparity + maxDisparity) > 0)
        {

            bilateralFilter(disp32F, filteredDisparity1, 9, 150, 150);
            medianBlur(filteredDisparity1, filteredDisparity2, 3);          // 中值滤波
            disparityMaps.push_back(filteredDisparity2);
        }
        else {
            // 获取矩阵的最小值和最大值
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(disp32F, &minVal, &maxVal, &minLoc, &maxLoc);

            
            Mat mask = Mat::zeros(disp32F.size(), CV_8UC1);                 // 创建掩膜矩阵，将与最小视差相等的像素位置设置为255
            compare(disp32F, minVal, mask, CMP_EQ);
            bitwise_not(mask, mask);                                        // 反转掩膜矩阵
            Mat maskedisp32F;                                               // 将掩膜应用到视差图上
            disp32F.copyTo(maskedisp32F, mask);
            maskedisp32F = -maskedisp32F;                                   // 视差图为负的取反


            maskedisp32F.setTo(-1, mask == 0);                              // 将之前掩膜掉的区域全赋值为-1

            bilateralFilter(maskedisp32F, filteredDisparity1, 9, 150, 150);
            medianBlur(filteredDisparity1, filteredDisparity2, 3);          // 中值滤波
            filteredDisparity2 = -filteredDisparity2;
            
            filteredDisparity2.setTo(minVal, mask == 0);                    // 再将之前掩膜掉的区域全赋值为最小视差
            disparityMaps.push_back(filteredDisparity2);
        }
        
        cout << "像对" << filenames[i] << "-" << filenames[j] << "密集匹配完成！" << endl;
    }
}

/**
 * @brief 输出视差图
 * @param disparityMaps 视差图
 * @param foldername 文件夹
 * @param filenames 文件名
*/
void write_dispimg(
    const vector<Mat> disparityMaps,
    const string foldername,
    const vector<string> filenames)
{
    for (int i = 0; i < imagenums; i++)
    {
        int j = i + 1;
        if (j == imagenums)
        {
            j = 0;
        }

        if (_access(foldername.c_str(), 0) == -1)                           //返回值为-1，表示不存在
        {
            cout << "创建文件夹" << foldername << endl;;
            _mkdir(foldername.c_str());
        }

        //视差图显示
        
        Mat disp8U = Mat(disparityMaps[i].rows, disparityMaps[i].cols, CV_8UC1);
       

        //disparityMaps[i].convertTo(disp8U, CV_8U, 255 / (nmDisparities * 16.));//转8位
        normalize(disparityMaps[i], disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
        imwrite(foldername + "/disparity_image_" + filenames[i] + "-" + filenames[j] + ".jpg", disp8U);
    }
    cout << "视差图保存成功！" << endl;
}

/**
 * @brief 生成3d点云
 * @param eplimgls 左核线影像
 * @param disparityMaps 视差图
 * @param extrinsics 相机外参
 * @param Qs 深度矩阵
 * @param r 核线到原图的旋转矩阵
 * @param foldername 文件夹
 * @param filenames 文件名
*/
void create_3dpoints(
    const vector<Mat> eplimgls,
    const vector<Mat> disparityMaps,
    const vector<Mat> extrinsics,
    const vector<Mat> Qs,
    const vector<Mat> r, 
    const string foldername,
    const vector<string> filenames)
{
    for (int i = 0; i < imagenums; i++)
    {
        int n = i + 1;
        if (n == imagenums)
        {
            n = 0;
        }

        Mat disp32F;
        Mat xyz;
        disparityMaps[i].convertTo(disp32F, CV_32FC1, 16);       // 计算坐标时乘以16
        reprojectImageTo3D(disp32F, xyz, Qs[i]);
        // 点云滤波
        // 统计滤波算法
        float threshold = 0.1; // 阈值
        int windowSize = 11; // 窗口大小
        Mat filteredCloud = xyz.clone();
        for (int y = windowSize / 2; y < xyz.rows - windowSize / 2; y++)
        {
            for (int x = windowSize / 2; x < xyz.cols - windowSize / 2; x++)
            {
                float meanZ = 0.0;
                int count = 0;

                // 计算窗口内的平均深度值
                for (int dy = -windowSize / 2; dy <= windowSize / 2; dy++)
                {
                    for (int dx = -windowSize / 2; dx <= windowSize / 2; dx++)
                    {
                        float depth = xyz.at<Vec3f>(y + dy, x + dx)[2];
                        if (depth > 0)
                        {
                            meanZ += depth;
                            count++;
                        }
                    }
                }

                if (count > 0)
                {
                    meanZ /= count;

                    // 基于阈值进行滤波
                    float currZ = xyz.at<Vec3f>(y, x)[2];
                    if (abs(currZ - meanZ) > threshold)
                    {
                        filteredCloud.at<Vec3f>(y, x)[2] = 0.0; // 将深度置为0
                    }
                }
            }
        }
        Mat XYZ = filteredCloud;
        for (int j = 0; j < eplimgls[i].rows; j++)
        {
            for (int k = 0; k < eplimgls[i].cols; k++)
            {           

                double x, y, z;
                x = XYZ.at<Vec3f>(j, k)[0];
                y = XYZ.at<Vec3f>(j, k)[1];
                z = XYZ.at<Vec3f>(j, k)[2];
                Mat realpoint3D = Mat(3, 1, CV_64FC1);
                Mat Eppoint3D = Mat(3, 1, CV_64FC1);
                Eppoint3D.at<double>(0, 0) = x * 16;
                Eppoint3D.at<double>(1, 0) = y * 16;
                Eppoint3D.at<double>(2, 0) = z * 16;

                realpoint3D = r[i].inv() * Eppoint3D;
                Mat R, T;
                R = extrinsics[i](Range(0, 3), Range(0, 3)).clone();
                T = extrinsics[i](Range(0, 3), Range(3, 4)).clone();


                Mat world_points;
                world_points = R.inv() * (realpoint3D - T);

                XYZ.at<Vec3f>(j, k)[0] = world_points.at<double>(0, 0);
                XYZ.at<Vec3f>(j, k)[1] = world_points.at<double>(1, 0);
                XYZ.at<Vec3f>(j, k)[2] = world_points.at<double>(2, 0);
            }
        }


        Mat disp8U = Mat(disparityMaps[i].rows, disparityMaps[i].cols, CV_8UC1);
     
        normalize(disparityMaps[i], disp8U, 0, 255, NORM_MINMAX, CV_8UC1);


        if (_access(foldername.c_str(), 0) == -1)//返回值为-1，表示不存在
        {
            cout << "创建文件夹" << foldername << endl;;
            _mkdir(foldername.c_str());
        }
        string filename = foldername + "/3Dcloud_pts_" + filenames[i] + "-" + filenames[n] + ".txt";
        ofstream fout(filename, ios::trunc | ios::app | ios::out);
        fout.open(filename);
        for (int j = 0; j < eplimgls[i].rows; j++)
        {
            for (int k = 0; k < eplimgls[i].cols; k++)
            {
             
                int gray;
                gray = disp8U.at<uchar>(j, k);


                int B, G, R;
                B = eplimgls[i].at<Vec3b>(j, k)[0];
                G = eplimgls[i].at<Vec3b>(j, k)[1];
                R = eplimgls[i].at<Vec3b>(j, k)[2];

                double x, y, z;
                x = XYZ.at<Vec3f>(j, k)[0];
                y = XYZ.at<Vec3f>(j, k)[1];
                z = XYZ.at<Vec3f>(j, k)[2];
                if (//(x > -22 && x < 22) && (y > -15 && y < 15) && (z > 33 && z < 39) &&
                    ((x > -1.5 && x < 1.5) && (y > -1.55 && y < 1.5) && (z > 3 && z < 10)) &&
                    (gray != 0)&&
                    (R != 0 && G != 0 && B != 0) &&
                    (!isnan(XYZ.at<Vec3f>(j, k)[0]))) {

                    fout << x << " ";
                    fout << y << " ";
                    fout << z << " ";
                    fout << R << " ";
                    fout << G << " ";
                    fout << B << endl;
                }
                else
                    continue;
            }
        }
        fout.close();
        cout << "点云" << i + 1 << "保存成功！" << endl;
    }
}

/**
 * @brief 格式化输出字符串
 * @param  str 输出内容
*/
void coutstring1(const string str)
{
    int width = 60;

    if (str.length() >= width) {
        cout << str << endl;
    }
    else {
        int padding = width - str.length();
        int leftPadding = padding / 2;
        int rightPadding = padding - leftPadding;

        cout << setw(leftPadding) << setfill('*') << "" << left << str;
        cout << setw(rightPadding) << "" << "*" << endl;
    }
}


void optimize_process(PointCloud<PointXYZRGB>::Ptr& cloud)
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

int main()
{
// 算法总耗时
    double total_count_beg = getTickCount();
    coutstring1("开始SFM");

// ******************************读取影像******************************

    cout << "<<====正在读取影像====<<" << endl;
    string folderPath = "data";                                           // 文件夹路径
    vector<Mat> images;                                                   // 存储图片的向量
    vector<string> filenames;                                             // 影像名称
    // --------------------------------
    create_imglines(folderPath, images, filenames);
    //imagenums = 6;
        // --------------------------------
        double t1 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double readimg_count = t1;
        cout << "读取影像耗时:  " << readimg_count << " (s) " << endl;
        cout << "总耗时:  " << t1 << " (s) " << endl;

// ***************************初始化相机参数****************************

    cout << "<<====初始化相机参数====<<" << endl;
    vector<Mat> exter_matrix;                                             // 相机外参
    vector<Mat> inner_matrix;                                             // 相机内参
    // --------------------------------
    init_params(exter_matrix, inner_matrix);
        // --------------------------------
        double t2 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double initparas_count = t2 - t1;
        cout << "初始化相机参数耗时:  " << initparas_count << " (s) " << endl;
        cout << "总耗时:  " << t2 << " (s) " << endl;
// *************************计算每一张图特征点**************************

    coutstring1("开始特征匹配");
    cout << "<<====正在提取特征点====<<" << endl;
    vector<vector<KeyPoint>> keypoints;                                   // 所有图片的特征点
    vector<Mat> descriptors;                                              // 所有图片的特征点描述子
    // --------------------------------
    cal_keypoints(images, 
        filenames,
        keypoints, descriptors);
        // --------------------------------
        double t3 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double cal_keypoints_count = t3 - t2;
        cout << "计算特征点耗时:  " << cal_keypoints_count << " (s) " << endl;
        cout << "总耗时:  " << t3 << " (s) " << endl;

// *************************两两之间进行特征匹配**************************
    
    cout << "<<====正在进行特征匹配====<<" << endl;
    vector<vector<DMatch>> matches;                                       // 相邻航片匹配结果
    vector<Mat> matchimg;                                                 // 匹配结果图
    // --------------------------------
    match(images,
        keypoints,
        descriptors, 
        inner_matrix, filenames, matches);
    writematchimg(images,
        matches, keypoints,
        "matches", filenames, matchimg);
        // --------------------------------
        double t4 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double match_count = t4 - t3;
        cout << "计算特征点耗时:  " << match_count << " (s) " << endl;
        cout << "总耗时:  " << t4 << " (s) " << endl;

// ***************************光束法区域网平差*****************************
    
    coutstring1("生成稀疏点云");
    cout << "<<====正在光束法区域网平差====<<" << endl;
    vector<vector<Point3d>> points3d;                                     // 稀疏重建3d点
    // --------------------------------
    bundle_adjustment(
        keypoints, matches, 
        exter_matrix, inner_matrix, points3d);
    writeCameraParams(
        inner_matrix, 
        exter_matrix, "bundle_adjustment", filenames);
    write_3dpts(
        points3d, "sparse_3Dpoints", filenames);
        // --------------------------------
        double t5 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double bundle_adjustment_count = t5 - t4;
        cout << "光束法区域网平差耗时:  " << bundle_adjustment_count << " (s) " << endl;
        cout << "总耗时:  " << t5 << " (s) " << endl;
        cout << ">>====稀疏重建完成====>>" << endl;

// *****************************生成核线影像*******************************
    
    coutstring1("开始密集重建");
    cout << "<<====正在生成核线影像====<<" << endl;
    vector<Mat> eplimgls, eplimgrs;                                       // 左右核线影像
    vector<Mat> Qs;                                                       // 深度矩阵
    vector<Mat> r1, r2;                                                   // 原图到核线影像的旋转矩阵
    vector<vector<KeyPoint>> epikpls, epikprs;                            // 核线影像特征点
    vector<int> mindisps, maxdisps;                                       // 最大最小视差
    // --------------------------------
    create_epimg(images,
        keypoints, matches,
        exter_matrix,
        inner_matrix,
        eplimgls, eplimgrs,
        Qs, r1, r2,
        epikpls, epikprs);
    cal_disp(
        epikpls,epikprs,
        mindisps, maxdisps);
    write_epipolar_image(
        eplimgls, eplimgrs,
        epikpls, epikprs,
        "epipolar_image", filenames);
        // --------------------------------
        double t6 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double create_epipolar_image_count = t6 - t5;
        cout << "生成核线影像耗时:  " << bundle_adjustment_count << " (s) " << endl;
        cout << "总耗时:  " << t6 << " (s) " << endl;

// *******************************密集匹配********************************

    cout << "<<====正在进行密集匹配====<<" << endl;
    vector<Mat> dispimgs;                                                  // 视差图
    // --------------------------------
    dense_match(
        eplimgls, eplimgrs,
        mindisps, maxdisps,
        filenames, dispimgs);
    write_dispimg(dispimgs,
        "disparity_image", filenames);
        // --------------------------------
        double t7 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double create_disps_count = t7 - t6;
        cout << "生成核线影像耗时:  " << create_disps_count << " (s) " << endl;
        cout << "总耗时:  " << t7 << " (s) " << endl;

// *****************************生成密集点云*******************************
    
    cout << "<<====正在生成密集点云====<<" << endl;
    // --------------------------------
    create_3dpoints(
        eplimgls,
        dispimgs,
        exter_matrix,
        Qs,
        r1,
        "dense_3Dpoints",
        filenames);
        // --------------------------------
        double t8 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double create_cpts_count = t8 - t7;
        cout << "生成密集点云耗时:  " << create_cpts_count << " (s) " << endl;
        cout << "总耗时:  " << t8 << " (s) " << endl;

// *****************************点云后处理*******************************

    cout << "<<====正在优化点云====<<" << endl;
    // --------------------------------
    string foldername = "dense_3Dpoints";
    boost::filesystem::path cptsPath(foldername);

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

    // 遍历文件夹中的文件，并按文件名顺序读取
    boost::filesystem::directory_iterator endItr;
    int i = 1;
    for (boost::filesystem::directory_iterator itr(cptsPath); itr != endItr; ++itr) {

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

    optimize_process(cloud);

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
        // --------------------------------
        double t9 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double optimize_count = t9 - t8;
        cout << "点云优化耗时:  " << optimize_count << " (s) " << endl;
        cout << "总耗时:  " << t9 << " (s) " << endl;
    return 0;
}