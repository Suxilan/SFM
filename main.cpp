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

// �������Ƭ����
int imagenums;

/**
 * @brief ��ȡӰ��
 * @param path Ӱ��·��
 * @return Mat��Ӱ��
*/
Mat readimg(string path)
{
    // ��ȡRGBͼ��
    Mat img = imread(path, IMREAD_COLOR);
    //���ͼ���Ƿ���سɹ�
    if (img.empty() || img.depth() != CV_8U)
    {
        CV_Error(CV_StsBadArg, "����ͼ��Ϊ�գ�����ͼ����Ȳ���CV_8U");
        return Mat();
    }

    return img;
}

/**
 * @brief ��������Ӱ��
 * @param folderPath �ļ���·��
 * @param images Mat��Ӱ��
 * @param filenames Ӱ������
*/
void create_imglines(
    const string folderPath,
    vector<Mat>& images,
    vector<string>& filenames)
{
    string pattern = folderPath + "/*.png";  // �����ļ�����ֻ�� jpg ��ʽ��ͼƬ
    vector<String> imagePaths;
    glob(pattern, imagePaths);

    // ����ͼƬ�ļ�����ȡ��������
    for (const auto& imagePath : imagePaths) {
        Mat image = readimg(imagePath);
        if (!image.empty()) {
            images.push_back(image);

            // ��ȡӰ���ļ�����ȥ����չ�����������
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
    cout << endl << "��Ƭ������" << imagenums << endl;
}

/**
 * @brief ��ʼ���������
 * @param exter_matrix ������
 * @param inner_matrix ����ڲ�
*/
void init_params(
    vector<Mat>& exter_matrix,
    vector<Mat>& inner_matrix)
{
    // �˴�����д�ı���ȡ
    // ...
    // ��ʼ��
    Mat origin_matrix = (Mat_<double>(4, 4) <<                      // ��һ��Ϊԭ��
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
    Mat intrinsic = (Mat_<double>(3, 3) << fx, 0, x0,                // ��ʼ��������Ƭ���ڲ�
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
 * @brief ����������
 * @param imgs Ӱ��
 * @param filenames Ӱ������
 * @param kpts ������
 * @param descriptors ������������
*/
void cal_keypoints(
    const vector<Mat> imgs,
    const vector<string> filenames,
    vector<vector<KeyPoint>> & kpts,
    vector<Mat> & descriptors)
{
    for (int i = 0; i < imagenums; i++)
    {
        cout << "���ڼ���� " << i + 1 << " ��Ӱ��: " << filenames[i];
        Mat img = imgs[i];
        vector<KeyPoint> kpt;
        Mat descriptor;
        // --------------------------------
        // sift�������,��ȡ1000��������
        Ptr<SIFT> detector = SIFT::create();
        detector->detectAndCompute(img, Mat(), kpt, descriptor);
        kpts.push_back(kpt);
        descriptors.push_back(descriptor);
        cout << "����: " << kpt.size() << endl;
    }
}


/**
 * @brief ����Ӱ�������ƥ��
 * @param kpts ������
 * @param descriptors ������������
 * @param inner_matrix ����ڲ�
 * @param filenames Ӱ������
 * @param matches ƥ�������������
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
        cout << "����ƥ�����: " << filenames[i] << "-" << filenames[j] << endl;
        Mat descriptorsL = descriptors[i];
        Mat descriptorsR = descriptors[j];
        vector<KeyPoint> keypointsL = keypoints[i];
        vector<KeyPoint> keypointsR = keypoints[j];
        Mat cameramatrix = inner_matrix[i];
        // =============================================================
        Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
        // ƥ��������
        vector<vector<DMatch>> matches;
        matcher->knnMatch(descriptorsL, descriptorsR, matches, 2);

        // ����ɸѡ
        vector<DMatch> goodMatches;
        vector<Point2d> matchpointsL, matchpointsR;
        double ratioThreshold = 0.7; // �趨һ����ֵ������ɸѡƥ���
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

        //// ��ѡ��ɸѡ���ƥ��
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


        cout << "ƥ�����:" << goodMatches.size() << "��" << endl;

        // RANSAC�޳��ֲ�
        vector<DMatch> selectMatches;
        vector<uchar> inliers(matchpointsL.size(), 0);
        Mat EssentialMat = findEssentialMat(matchpointsL, matchpointsR, cameramatrix, FM_RANSAC, 0.99, 1.0, inliers);
        for (size_t k = 0; k < goodMatches.size(); k++) {
            if (inliers[k] != 0)
            {
                selectMatches.push_back(goodMatches[k]);
            }
        }
        cout << "ransacƥ��ɹ���" << selectMatches.size() << "��" << endl;
 
        //// ���ϵ��ɸѡ
        //vector<DMatch> correlationFilteredMatches;
        //double correlationThreshold = 0.9; // ���ϵ������ֵ
        //Size windowSize(21, 21); // ���ڴ�СΪ21x21
        //int border = windowSize.width / 2; // �߽��С�������жϴ����Ƿ�Խ��
        //for (const DMatch& match : selectMatches)
        //{
        //    Point2f ptL = keypointsL[match.queryIdx].pt;
        //    Point2f ptR = keypointsR[match.trainIdx].pt;

        //    // �߽���
        //    if (ptL.x < border || ptL.x >= img[i].cols - border ||
        //        ptL.y < border || ptL.y >= img[i].rows - border ||
        //        ptR.x < border || ptR.x >= img[j].cols - border ||
        //        ptR.y < border || ptR.y >= img[j].rows - border)
        //    {
        //        continue; // ����Խ��ĵ�
        //    }

        //    // ���㴰���ڵ����ϵ��
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
        //cout << "���ϵ��ɸѡ��ƥ�������" << correlationFilteredMatches.size() << "��" << endl;

        allmatches.push_back(selectMatches);
    }
}

/**
 * @brief ����ƴ��Ӱ��
 * @param imageL ����
 * @param imageR ����
 * @param cornersL ����ǵ�
 * @param cornersR ����ǵ�
 * @return ƴ��Ӱ��
*/
Mat drawmatches(
    Mat imageL, 
    Mat imageR, 
    vector<KeyPoint>cornersL, 
    vector<KeyPoint>cornersR)
{
    // ���ƽǵ�
    Mat CornerL_Detection;
    drawKeypoints(imageL, cornersL, CornerL_Detection, Scalar(0, 255, 255));
    Mat CornerR_Detection;
    drawKeypoints(imageR, cornersR, CornerR_Detection, Scalar(0, 255, 255));

    //��Ӱ��ƴ����һ��
    // ����ƴ�Ӻ��Ӱ��ߴ�
    int max_width = CornerL_Detection.cols + CornerR_Detection.cols;
    int max_height = max(CornerL_Detection.rows, CornerR_Detection.rows);

    Mat mergeline(max_height, max_width, CornerL_Detection.type());
    mergeline.setTo(Scalar::all(0)); // ��0���
    // ����Ӱ��ƴ�Ӻ��Ӱ����
    Mat roi1(mergeline, Rect(0, 0, CornerL_Detection.cols, CornerL_Detection.rows));
    CornerL_Detection.copyTo(roi1);
    Mat roi2(mergeline, Rect(CornerL_Detection.cols, 0, CornerR_Detection.cols, CornerR_Detection.rows));
    CornerR_Detection.copyTo(roi2);

    // ���ͼ��ͨ����
    if (mergeline.channels() == 1)
    {
        // ͼ��Ϊ�Ҷ�ͼ�񣬽���ͨ���任
        cvtColor(mergeline, mergeline, COLOR_GRAY2BGR);
    }
    //���������
    // ��ȡ��ǰʱ��
    time_t now = time(0);
    // �õ�ǰʱ����Ϊ��������ʼ�������������
    RNG rng(now);
    for (int i = 0; i < cornersL.size(); i++)
    {
        line(mergeline, Point(cornersL[i].pt.x, cornersL[i].pt.y), Point(cornersR[i].pt.x + imageL.cols, cornersR[i].pt.y), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, 0);
    }
    // ��ʾ���
    return mergeline;
}

/**
 * @brief ���ƥ��Ӱ��
 * @param imgs ԭͼ
 * @param allmatches ƥ������
 * @param keypoints ������
 * @param foldername ƥ��ͼ�ļ���
 * @param filenames �ļ���
 * @param matchimg ƥ��ͼ
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
            // ��ȡƥ��Ե������;���
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
            //����ֵΪ-1����ʾ������
        {
            _mkdir(foldername.c_str());
        }
        imwrite(foldername+"/Matches" + matchids + ".jpg", matchedImage);
        //waitKey(0);
    }
}

/**
 * @brief ������������ƽ��
 * @param keypoints ÿ����Ƭ��������
 * @param matches ������ƥ������
 * @param exter_matrix ������
 * @param inner_matrix ����ڲ�
 * @param pts_3d ��Ӧ�﷽������
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

    // ǰ��������Զ�����RT
    vector<KeyPoint>kp1s, kp2s;
    kp1s.resize(matches[0].size());
    kp2s.resize(matches[0].size());
    for (size_t j = 0; j < kp1s.size(); ++j)
    {
        // ��ȡƥ��Ե������;���
        kp1s[j] = keypoints[0][matches[0][j].queryIdx];
        kp2s[j] = keypoints[1][matches[0][j].trainIdx];
    }

    // ��Զ���
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
    Tr = Tl + Rl * T0;                                          // ���򵽵�һ��Ӱ������ϵ��
    // �õ���ʼRT����ʽ��ʼǰ������󷽽�������
    // �ܽ�Ϊ:
    // 1.�õ�RT;
    // 2.ǰ��������3D��;
    // 3.��������ͶӰ
    // 4.�󷽽���

    for (size_t i = 0; i < imagenums - 1; ++i)
    {
        int j = i + 2;
        if (j == imagenums)
        {
            j = 0;
        }
        cout << "�������" << i + 1 << endl;
        vector<KeyPoint>kps1, kps2, kps3;
        // ǰ��������Ҫ����,ǰ���ŵĵ��ڵ������ϳ���
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
        cout << "�μӹ������ĵ�����" << kps1.size() << endl;
        vector<Point3d>pt_3d;                                        // ǰ�������3D������
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
        Rodrigues(Rr, rvec);                                         // ����ת���� R ת��Ϊŷ������ת����
        // ��ȡ���
        camera_params.insert(camera_params.end(), rvec.val, rvec.val + 3);
        camera_params.insert(camera_params.end(), Tr.begin<double>(), Tr.end<double>());
        // ��ȡ�ڲ�
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(0, 0));
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(1, 1));
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(0, 2));
        camera_matrix.push_back(inner_matrix[i + 1].at<double>(1, 2));
        Bundle_adjustment(kps1, kps2, camera_params, camera_matrix, point3d);

        //�õ�ƽ������ξ���
        copy(camera_params.begin(), camera_params.begin() + 3, rvec.val);
        copy(camera_params.begin() + 3, camera_params.end(), Tr.ptr<double>());
        Rodrigues(rvec, Rr);
        Mat extrinsic = Mat::eye(4, 4, CV_64F);                     // ������ξ��� extrinsic
        Mat R_roi = extrinsic(Rect(0, 0, 3, 3));
        Rr.copyTo(R_roi);
        Mat T_roi = extrinsic(Rect(3, 0, 1, 3));
        Tr.copyTo(T_roi);
        exter_matrix[i + 1] = extrinsic;      
        Rl = Rr; Tl = Tr;                                           // ��θ�Ϊ��һ��Ӱ�����
        //�õ�ƽ�����ڲξ���
        Mat intrinsic = Mat::eye(3, 3, CV_64F);                     // ������ξ��� intrinsic
        intrinsic.at<double>(0, 0) = camera_matrix[0];
        intrinsic.at<double>(1, 1) = camera_matrix[1];
        intrinsic.at<double>(0, 2) = camera_matrix[2];
        intrinsic.at<double>(1, 2) = camera_matrix[3];
        inner_matrix[i + 1] = intrinsic;

        //�õ�ƽ����3d����
        for (size_t j = 0; j < point3d.size(); ++j)
        {
            pt_3d[j].x = point3d[j][0];
            pt_3d[j].y = point3d[j][1];
            pt_3d[j].z = point3d[j][2];
        }
        pts_3d.push_back(pt_3d);
        
        //��ʼ�󷽽���

        Mat rvec1;  // ��ת����
        Mat tvec1;  // ƽ������
  
        Resection(inner_matrix[j], kps3, pt_3d, rvec1, tvec1);
        Mat R; // ��ת����
        Rodrigues(rvec1, R);
        Rr = R; Tr = tvec1;
 
        // �ػ�
        if (i == imagenums - 2)
        {
            cout << "�������" << i + 2 << endl;
            vector<KeyPoint>kps1, kps2;
            for (size_t k = 0; k < matches[i + 1].size(); ++k)
            {
                KeyPoint kp1, kp2;                      // �������
                kp1 = keypoints[i + 1][matches[i + 1][k].queryIdx];
                kp2 = keypoints[j][matches[i + 1][k].trainIdx];
                kps1.push_back(kp1); kps2.push_back(kp2);
            }
  
            vector<Point3d>pt_3d;                                    // ǰ�������3D������
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
            Rodrigues(Rr, rvec);                                         // ����ת���� R ת��Ϊŷ������ת����
            // ��ȡ���
            camera_params.insert(camera_params.end(), rvec.val, rvec.val + 3);
            camera_params.insert(camera_params.end(), Tr.begin<double>(), Tr.end<double>());
            // ��ȡ�ڲ�
            camera_matrix.push_back(inner_matrix[j].at<double>(0, 0));
            camera_matrix.push_back(inner_matrix[j].at<double>(1, 1));
            camera_matrix.push_back(inner_matrix[j].at<double>(0, 2));
            camera_matrix.push_back(inner_matrix[j].at<double>(1, 2));
            Bundle_adjustment(kps1, kps2, camera_params, camera_matrix, point3d);

            //�õ�ƽ������ξ���
            copy(camera_params.begin(), camera_params.begin() + 3, rvec.val);
            copy(camera_params.begin() + 3, camera_params.end(), Tr.ptr<double>());
            Rodrigues(rvec, Rr);
            Mat extrinsic = Mat::eye(4, 4, CV_64F);                     // ������ξ��� extrinsic
            Mat R_roi = extrinsic(Rect(0, 0, 3, 3));
            Rr.copyTo(R_roi);
            Mat T_roi = extrinsic(Rect(3, 0, 1, 3));
            Tr.copyTo(T_roi);
            exter_matrix[j] = extrinsic;

            //�õ�ƽ�����ڲξ���
            Mat intrinsic = Mat::eye(3, 3, CV_64F);                     // ������ξ��� intrinsic
            intrinsic.at<double>(0, 0) = camera_matrix[0];
            intrinsic.at<double>(1, 1) = camera_matrix[1];
            intrinsic.at<double>(0, 2) = camera_matrix[2];
            intrinsic.at<double>(1, 2) = camera_matrix[3];
            inner_matrix[j] = intrinsic;

            //�õ�ƽ����3d����
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
 * @brief д���������
 * @param camera_intrinsics ����ڲ�
 * @param camera_extrinsics ������
 * @param foldername �ļ�������
 * @param filenames �ļ���
*/
void writeCameraParams(
    const vector<Mat>& camera_intrinsics,
    const vector<Mat>& camera_extrinsics,
    const string foldername,
    const vector<string> filenames) 
{

    if (_access(foldername.c_str(), 0) == -1)
        //����ֵΪ-1����ʾ������
    {
        _mkdir(foldername.c_str());
    }

    // ����������ֶο�Ⱥ;���
    const int field_width = 15;
    const int precision = 6;

    // ѭ������ÿ��������ڲκ����
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
    cout << "д������ɹ���" << endl;
}

/**
 * @brief д��ϡ���﷽������
 * @param pts_3d �﷽��
 * @param foldername �ļ���
 * @param filenames �ļ���
*/
void write_3dpts(
    const vector<vector<Point3d>> pts_3d,
    const string foldername,
    const vector<string> filenames)
{
    if (_access(foldername.c_str(), 0) == -1)
        //����ֵΪ-1����ʾ������
    {
        _mkdir(foldername.c_str());
    }

    // ����������ֶο�Ⱥ;���
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
    cout << "ϡ���﷽�㱣��ɹ���" << endl;
}

/**
 * @brief �����������
 * @param imgs ԭʼͼ��
 * @param keypoints ÿ��ͼ������
 * @param matches ������ƥ������
 * @param extrinsics ���
 * @param intrinsics �ڲ�
 * @param eplimgls �����Ӱ��
 * @param eplimgrs �Һ���Ӱ��
 * @param Qs ��Ⱦ���
 * @param img2epls ����任��ת����
 * @param img2eprs ����任��ת����
 * @param epikpls �����Ӱ��������
 * @param epikprs �Һ���Ӱ��������
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

        //��ȡÿ��ͼ��ľ���R,T
        Mat RTL, RTR;                                         //����ͼƬ���
        Mat RT;                                               //����ͼ����Ե�RT
        RTL = extrinsics[i]; RTR = extrinsics[j];
        RT = RTR * (RTL.inv());

        Mat rvecsMat, tvecsMat;                               //��Ե���ת�����ƽ�ƾ���
        rvecsMat = RT(Range(0, 3), Range(0, 3)).clone();
        tvecsMat = RT(Range(0, 3), Range(3, 4)).clone();

        // �ڲ�
        Mat intrinsicl = intrinsics[i];
        Mat intrinsicr = intrinsics[j];
        Mat distCoeffsL = Mat::zeros(1, 5, CV_64FC1);
        Mat distCoeffsR = Mat::zeros(1, 5, CV_64FC1);         // �������5������ϵ����k1,k2,p1,p2,k3
        
        Size image_size = imgl.size();                        // ͼ��ߴ�
        Mat dstL, dstR;
        Mat RvecsL(3, 3, CV_64FC1), RvecsR(3, 3, CV_64FC1);   // ����У�������ת����
        Mat PvecsL(3, 4, CV_64FC1), PvecsR(3, 4, CV_64FC1);   // ����У�����ӳ�����
        Mat Qvecs(4, 4, CV_64FC1);                            // ����У�������Ⱦ���
        stereoRectify(intrinsicl, distCoeffsL,
            intrinsicr, distCoeffsR,
            image_size, rvecsMat, tvecsMat,
            RvecsL, RvecsR,
            PvecsL, PvecsR, Qvecs, CALIB_ZERO_DISPARITY, -1);

        Qs.push_back(Qvecs);
        img2epls.push_back(RvecsL); 
        img2eprs.push_back(RvecsR);

        Mat rmapLfirst, rmapLsec;                             //��Ƭ������ӳ��任
        Mat rmapRfirst, rmapRsec;                             //��Ƭ������ӳ��任
        Mat imgEPL = Mat(imgl.rows, imgl.cols, CV_8UC3);      //�任�������ͼ��
        Mat imgEPR = Mat(imgr.rows, imgr.cols, CV_8UC3);      //�任�������ͼ��

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

        vector<KeyPoint> kpL_corrected, kpR_corrected;                  // ����У����Ĺؼ��㼯��
        kpL_corrected.resize(leftKeypoints.size());
        kpR_corrected.resize(rightKeypoints.size());
        // ����ԭʼͼ���ϵĹؼ����������
        Mat kpL_coords(leftKeypoints.size(), 1, CV_32FC2);
        Mat kpR_coords(rightKeypoints.size(), 1, CV_32FC2);

        // ��ԭʼͼ���ϵĹؼ���������䵽������
        for (int i = 0; i < leftKeypoints.size(); ++i) {
            kpL_coords.at<Vec2f>(i, 0) = Vec2f(leftKeypoints[i].pt.x, leftKeypoints[i].pt.y);
            kpR_coords.at<Vec2f>(i, 0) = Vec2f(rightKeypoints[i].pt.x, rightKeypoints[i].pt.y);
        }

        // ����У����Ĺؼ����������
        Mat kpL_corrected_coords, kpR_corrected_coords;

        // ʹ�þ����������У��
        undistortPoints(kpL_coords, kpL_corrected_coords, intrinsicl, distCoeffsL, RvecsL, PvecsL);
        undistortPoints(kpR_coords, kpR_corrected_coords, intrinsicr, distCoeffsR, RvecsR, PvecsR);

        // ��У����Ĺؼ�������ת��Ϊ�ؼ������
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
 * @brief �������Ӱ���Ӳ�
 * @param leftKeypoints ����������
 * @param rightKeypoints ����������
 * @param mindisps ��С�Ӳ�
 * @param maxdisps ����Ӳ�
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
        // ������С������Ӳ�ֵ
        double maxDisparity = -100000.0;
        double minDisparity = 100000.0;
        vector<KeyPoint> kpL, kpR;
        for (size_t k = 0; k < leftKeypoints.size(); ++k)
        {
            // �����ӲX�᷽���ϵ�λ�ƣ�
            double disparity = leftKeypoints[k].pt.x - rightKeypoints[k].pt.x;

            // ��������Ӳ�ֵ
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
 * @brief д�����Ӱ��
 * @param eplimgls �����Ӱ��
 * @param eplimgrs �Һ���Ӱ��
 * @param epikpls �����Ӱ��������
 * @param epikprs �Һ���Ӱ��������
 * @param foldername �ļ���
 * @param filenames �ļ���
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
        // ����ƴ�Ӻ��ͼ��
        Mat stitchedImage = drawmatches(eplimgls[i], eplimgrs[i], epikpls[i], epikprs[i]);

        if (_access(foldername.c_str(), 0) == -1)//����ֵΪ-1����ʾ������
        {
            cout << "�����ļ���" << foldername << endl;;
            _mkdir(foldername.c_str());
        }

        imwrite(foldername + "/epipolar_image_" + filenames[i]+ "-" + filenames[j] + ".jpg", stitchedImage);
    }
    cout << "����Ӱ�񱣴�ɹ���" << endl;
}

/**
 * @brief �ܼ�ƥ��
 * @param eplimgls �����Ӱ��
 * @param eplimgrs �Һ���Ӱ��
 * @param mindisps ��С�Ӳ�
 * @param maxdisps ����Ӳ�
 * @param filenames �ļ���
 * @param disparityMaps �Ӳ�ͼ
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
        cout << "��С�Ӳ�Ϊ��" << minDisparity << endl;
        cout << "����Ӳ�Ϊ��" << maxDisparity << endl;

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
       
 
        int nmDisparities = (abs(maxDisparity - minDisparity) + 15) & -16;  //�Ӳ�������Χ
        int pngChannels = grayL.channels();                                 //��ȡ����ͼͨ����
        int winSize = 3;

        Ptr<StereoSGBM> sgbm = cv::StereoSGBM::create();
        // ����SGBM����
        sgbm->setPreFilterCap(31);                                          //Ԥ�����˲����ض�ֵ
        sgbm->setBlockSize(winSize);                                        //SAD���ڴ�С
        sgbm->setP1(8 * pngChannels * winSize * winSize);                   //�����Ӳ�ƽ���ȵ�һ����
        sgbm->setP2(32 * pngChannels * winSize * winSize);                  //�����Ӳ�ƽ���ȵڶ�����
        sgbm->setMinDisparity(minDisparity);                                //��С�Ӳ�
        sgbm->setNumDisparities(nmDisparities);                             //�Ӳ�������Χ
        sgbm->setUniquenessRatio(10);                                       //�Ӳ�Ψһ�԰ٷֱ�
        sgbm->setSpeckleWindowSize(64);                                     //ɢ�ߴ��ڴ�С
        sgbm->setSpeckleRange(2);                                           //ɢ�߷�Χ������ʽ���� 16
        sgbm->setDisp12MaxDiff(1);                                          //�����Ӳ�ͼ����������,����һ����
        sgbm->setMode(StereoSGBM::MODE_HH);                                 //����ȫ�ߴ�˫ͨ����̬����㷨
        Mat disparityMap;
        sgbm->compute(grayL, grayR, disparityMap);

        Mat disp32F = Mat(disparityMap.rows, disparityMap.cols, CV_32FC1);
        disparityMap.convertTo(disp32F, CV_32FC1, 1.0 / 16);                //����16�õ���ʵ�Ӳ�
        // �Ӳ�ͼ����
        Mat filteredDisparity1 = Mat::zeros(disp32F.size(), disp32F.type());
        Mat filteredDisparity2 = Mat::zeros(disp32F.size(), disp32F.type());

        // ˫���˲���ѹ�Ƶ�Ƶ��Ϣ���ص㣬���Ӳ�ͼΪ����Ҫȡ������
        if ((minDisparity + maxDisparity) > 0)
        {

            bilateralFilter(disp32F, filteredDisparity1, 9, 150, 150);
            medianBlur(filteredDisparity1, filteredDisparity2, 3);          // ��ֵ�˲�
            disparityMaps.push_back(filteredDisparity2);
        }
        else {
            // ��ȡ�������Сֵ�����ֵ
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(disp32F, &minVal, &maxVal, &minLoc, &maxLoc);

            
            Mat mask = Mat::zeros(disp32F.size(), CV_8UC1);                 // ������Ĥ���󣬽�����С�Ӳ���ȵ�����λ������Ϊ255
            compare(disp32F, minVal, mask, CMP_EQ);
            bitwise_not(mask, mask);                                        // ��ת��Ĥ����
            Mat maskedisp32F;                                               // ����ĤӦ�õ��Ӳ�ͼ��
            disp32F.copyTo(maskedisp32F, mask);
            maskedisp32F = -maskedisp32F;                                   // �Ӳ�ͼΪ����ȡ��


            maskedisp32F.setTo(-1, mask == 0);                              // ��֮ǰ��Ĥ��������ȫ��ֵΪ-1

            bilateralFilter(maskedisp32F, filteredDisparity1, 9, 150, 150);
            medianBlur(filteredDisparity1, filteredDisparity2, 3);          // ��ֵ�˲�
            filteredDisparity2 = -filteredDisparity2;
            
            filteredDisparity2.setTo(minVal, mask == 0);                    // �ٽ�֮ǰ��Ĥ��������ȫ��ֵΪ��С�Ӳ�
            disparityMaps.push_back(filteredDisparity2);
        }
        
        cout << "���" << filenames[i] << "-" << filenames[j] << "�ܼ�ƥ����ɣ�" << endl;
    }
}

/**
 * @brief ����Ӳ�ͼ
 * @param disparityMaps �Ӳ�ͼ
 * @param foldername �ļ���
 * @param filenames �ļ���
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

        if (_access(foldername.c_str(), 0) == -1)                           //����ֵΪ-1����ʾ������
        {
            cout << "�����ļ���" << foldername << endl;;
            _mkdir(foldername.c_str());
        }

        //�Ӳ�ͼ��ʾ
        
        Mat disp8U = Mat(disparityMaps[i].rows, disparityMaps[i].cols, CV_8UC1);
       

        //disparityMaps[i].convertTo(disp8U, CV_8U, 255 / (nmDisparities * 16.));//ת8λ
        normalize(disparityMaps[i], disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
        imwrite(foldername + "/disparity_image_" + filenames[i] + "-" + filenames[j] + ".jpg", disp8U);
    }
    cout << "�Ӳ�ͼ����ɹ���" << endl;
}

/**
 * @brief ����3d����
 * @param eplimgls �����Ӱ��
 * @param disparityMaps �Ӳ�ͼ
 * @param extrinsics ������
 * @param Qs ��Ⱦ���
 * @param r ���ߵ�ԭͼ����ת����
 * @param foldername �ļ���
 * @param filenames �ļ���
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
        disparityMaps[i].convertTo(disp32F, CV_32FC1, 16);       // ��������ʱ����16
        reprojectImageTo3D(disp32F, xyz, Qs[i]);
        // �����˲�
        // ͳ���˲��㷨
        float threshold = 0.1; // ��ֵ
        int windowSize = 11; // ���ڴ�С
        Mat filteredCloud = xyz.clone();
        for (int y = windowSize / 2; y < xyz.rows - windowSize / 2; y++)
        {
            for (int x = windowSize / 2; x < xyz.cols - windowSize / 2; x++)
            {
                float meanZ = 0.0;
                int count = 0;

                // ���㴰���ڵ�ƽ�����ֵ
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

                    // ������ֵ�����˲�
                    float currZ = xyz.at<Vec3f>(y, x)[2];
                    if (abs(currZ - meanZ) > threshold)
                    {
                        filteredCloud.at<Vec3f>(y, x)[2] = 0.0; // �������Ϊ0
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


        if (_access(foldername.c_str(), 0) == -1)//����ֵΪ-1����ʾ������
        {
            cout << "�����ļ���" << foldername << endl;;
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
        cout << "����" << i + 1 << "����ɹ���" << endl;
    }
}

/**
 * @brief ��ʽ������ַ���
 * @param  str �������
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

int main()
{
// �㷨�ܺ�ʱ
    double total_count_beg = getTickCount();
    coutstring1("��ʼSFM");

// ******************************��ȡӰ��******************************

    cout << "<<====���ڶ�ȡӰ��====<<" << endl;
    string folderPath = "data";                                           // �ļ���·��
    vector<Mat> images;                                                   // �洢ͼƬ������
    vector<string> filenames;                                             // Ӱ������
    // --------------------------------
    create_imglines(folderPath, images, filenames);
    //imagenums = 6;
        // --------------------------------
        double t1 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double readimg_count = t1;
        cout << "��ȡӰ���ʱ:  " << readimg_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t1 << " (s) " << endl;

// ***************************��ʼ���������****************************

    cout << "<<====��ʼ���������====<<" << endl;
    vector<Mat> exter_matrix;                                             // ������
    vector<Mat> inner_matrix;                                             // ����ڲ�
    // --------------------------------
    init_params(exter_matrix, inner_matrix);
        // --------------------------------
        double t2 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double initparas_count = t2 - t1;
        cout << "��ʼ�����������ʱ:  " << initparas_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t2 << " (s) " << endl;
// *************************����ÿһ��ͼ������**************************

    coutstring1("��ʼ����ƥ��");
    cout << "<<====������ȡ������====<<" << endl;
    vector<vector<KeyPoint>> keypoints;                                   // ����ͼƬ��������
    vector<Mat> descriptors;                                              // ����ͼƬ��������������
    // --------------------------------
    cal_keypoints(images, 
        filenames,
        keypoints, descriptors);
        // --------------------------------
        double t3 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double cal_keypoints_count = t3 - t2;
        cout << "�����������ʱ:  " << cal_keypoints_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t3 << " (s) " << endl;

// *************************����֮���������ƥ��**************************
    
    cout << "<<====���ڽ�������ƥ��====<<" << endl;
    vector<vector<DMatch>> matches;                                       // ���ں�Ƭƥ����
    vector<Mat> matchimg;                                                 // ƥ����ͼ
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
        cout << "�����������ʱ:  " << match_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t4 << " (s) " << endl;

// ***************************������������ƽ��*****************************
    
    coutstring1("����ϡ�����");
    cout << "<<====���ڹ�����������ƽ��====<<" << endl;
    vector<vector<Point3d>> points3d;                                     // ϡ���ؽ�3d��
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
        cout << "������������ƽ���ʱ:  " << bundle_adjustment_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t5 << " (s) " << endl;
        cout << ">>====ϡ���ؽ����====>>" << endl;

// *****************************���ɺ���Ӱ��*******************************
    
    coutstring1("��ʼ�ܼ��ؽ�");
    cout << "<<====�������ɺ���Ӱ��====<<" << endl;
    vector<Mat> eplimgls, eplimgrs;                                       // ���Һ���Ӱ��
    vector<Mat> Qs;                                                       // ��Ⱦ���
    vector<Mat> r1, r2;                                                   // ԭͼ������Ӱ�����ת����
    vector<vector<KeyPoint>> epikpls, epikprs;                            // ����Ӱ��������
    vector<int> mindisps, maxdisps;                                       // �����С�Ӳ�
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
        cout << "���ɺ���Ӱ���ʱ:  " << bundle_adjustment_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t6 << " (s) " << endl;

// *******************************�ܼ�ƥ��********************************

    cout << "<<====���ڽ����ܼ�ƥ��====<<" << endl;
    vector<Mat> dispimgs;                                                  // �Ӳ�ͼ
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
        cout << "���ɺ���Ӱ���ʱ:  " << create_disps_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t7 << " (s) " << endl;

// *****************************�����ܼ�����*******************************
    
    cout << "<<====���������ܼ�����====<<" << endl;
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
        cout << "�����ܼ����ƺ�ʱ:  " << create_cpts_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t8 << " (s) " << endl;

// *****************************���ƺ���*******************************

    cout << "<<====�����Ż�����====<<" << endl;
    // --------------------------------
    string foldername = "dense_3Dpoints";
    boost::filesystem::path cptsPath(foldername);

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

    // �����ļ����е��ļ��������ļ���˳���ȡ
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
        cout << "���ƶ�ȡ�ɹ�!���������� " << cloud->size() << endl;

        i++;
    }

    optimize_process(cloud);

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
        // --------------------------------
        double t9 = ((double)getTickCount() - total_count_beg) / getTickFrequency();
        double optimize_count = t9 - t8;
        cout << "�����Ż���ʱ:  " << optimize_count << " (s) " << endl;
        cout << "�ܺ�ʱ:  " << t9 << " (s) " << endl;
    return 0;
}