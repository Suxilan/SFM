#pragma once
#ifndef BUNDLEADJUSTMENT_H
#define BUNDLEADJUSTMENT_H
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

struct ProjectionResidual {
    ProjectionResidual(const KeyPoint _observed_keypoint)
        : observed_keypoint(_observed_keypoint) {}

    template <typename T>
    bool operator()(
        const T* const extrinsic, 
        const T* const intrinsic,
        const T* const point, 
        T* residuals) const {
        // 外参参数
        const T* const rotation = extrinsic;
        const T* const translation = extrinsic + 3;


        // 读取相机内参
        const T* fx = intrinsic;
        const T* fy = intrinsic + 1;
        const T* cx = intrinsic + 2;
        const T* cy = intrinsic + 3;

        // 投影方程
        T pos_proj[3];
        ceres::AngleAxisRotatePoint(rotation, point, pos_proj);
        pos_proj[0] += translation[0];
        pos_proj[1] += translation[1];
        pos_proj[2] += translation[2];
        const T xp = pos_proj[0] / pos_proj[2];
        const T yp = pos_proj[1] / pos_proj[2];

        // 转换到图像坐标系下
        const T u = fx[0] * xp + cx[0];
        const T v = fy[0] * yp + cy[0];

        // 计算重投影误差
        residuals[0] = u - T(observed_keypoint.pt.x);
        residuals[1] = v - T(observed_keypoint.pt.y);

        return true;
    }

    static ceres::CostFunction* Create(const KeyPoint _observed_keypoint) {
        return new ceres::AutoDiffCostFunction<ProjectionResidual, 2, 6, 4, 3>(
            new ProjectionResidual(_observed_keypoint));
    }

    KeyPoint observed_keypoint;
};


// 相对定向，求R、T
void Pose_estimation(
    const Mat cameraMatrix,
    const vector<KeyPoint> kpl,
    const vector<KeyPoint> kpr,
    Mat& R, Mat& T)
{
    vector<Point2d> pointsL, pointsR;
    for (size_t i = 0; i < kpl.size(); i++)
    {
        pointsL.push_back(kpl[i].pt);
        pointsR.push_back(kpr[i].pt);
    }
    Mat EssentialMat = findEssentialMat(pointsL, pointsR, cameraMatrix, RANSAC);
    recoverPose(EssentialMat, pointsL, pointsR, cameraMatrix, R, T);
}

// 像素坐标转相机坐标
Point2d pixel2cam(const Point2d& p, const Mat& K)
{
    return Point2d
    (
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

// 前方交会
void Triangulation(
    const Mat cameraMatrix1,
    const Mat cameraMatrix2,
    const vector<KeyPoint> kpl,
    const vector<KeyPoint> kpr,
    const Mat r, const Mat t,
    const Mat R, const Mat T,
    vector<Point3d>& points_3d)
{
    Mat T1 = (Mat_<double>(3, 4) <<
        r.at<double>(0, 0), r.at<double>(0, 1), r.at<double>(0, 2), t.at<double>(0, 0),
        r.at<double>(1, 0), r.at<double>(1, 1), r.at<double>(1, 2), t.at<double>(1, 0),
        r.at<double>(2, 0), r.at<double>(2, 1), r.at<double>(2, 2), t.at<double>(2, 0));

    Mat T2 = (Mat_<double>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2, 0));
    vector<Point2d> pts1, pts2;

    // 像素坐标转换到归一化相机坐标
    for (size_t i = 0; i < kpl.size(); i++)
    {
        pts1.push_back(pixel2cam(kpl[i].pt, cameraMatrix1));
        pts2.push_back(pixel2cam(kpr[i].pt, cameraMatrix2));
    }

    Mat pts_4d;//齐次坐标
    triangulatePoints(T1, T2, pts1, pts2, pts_4d);

    for (int i = 0; i < pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<double>(3, 0);
        Point3d p(
            x.at<double>(0, 0),
            x.at<double>(1, 0),
            x.at<double>(2, 0)
        );
        points_3d.push_back(p);
    }
}


// 后方交会
void Resection(
    const Mat cameraMatrix,
    const vector<KeyPoint> kpts,
    const vector<Point3d> points_3d,
    Mat& r, Mat& t)
{
    Mat distortion_coeffs = Mat::zeros(1, 5, CV_64F);
    vector<Point2d> pts;

    // 像素坐标转换到归一化相机坐标
    for (size_t i = 0; i < kpts.size(); i++)
    {
        pts.push_back(kpts[i].pt);
    }

    solvePnPRansac(points_3d, pts, cameraMatrix, distortion_coeffs, r, t, false, SOLVEPNP_EPNP);
}


void Bundle_adjustment(
    const vector<KeyPoint> kpL,
    const vector<KeyPoint> kpR,
    vector<double>& extrinsic,
    vector<double>& intrinsic,
    vector<vector<double>>& points_3d)
{
    ceres::Problem problem;
    ceres::LossFunction* lossfuction = new ceres::CauchyLoss(1);
   
    // 添加残差块
    for (int i = 0; i < kpL.size(); ++i) {
        const KeyPoint& observed_keypoint = kpR[i];  // 右影像的匹配点
        
        problem.AddParameterBlock(extrinsic.data(), 6);    // 相机外参有6个参数
        problem.AddParameterBlock(intrinsic.data(), 4);    // 相机内参有4个参数
        problem.AddParameterBlock(points_3d[i].data(), 3);             // 3D点坐标有3个参数

        ceres::CostFunction* cost_function = ProjectionResidual::Create(observed_keypoint);
        problem.AddResidualBlock(cost_function, lossfuction, extrinsic.data(), intrinsic.data(), points_3d[i].data());

    }
    // 设置优化参数
    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //options.linear_solver_type = ceres::SPARSE_SCHUR;
    //options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = true;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 打印优化结果
    if (!summary.IsSolutionUsable())
    {
        cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        //cout << summary.FullReport() << std::endl;
        // Display statistics about the minimization
        //cout << endl
        //    << "Bundle Adjustment statistics (approximated RMSE):\n"
        //    << " #views: " << extrinsics.size() << "\n"
        //    << " #residuals: " << summary.num_residuals << "\n"
        //    << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
        //    << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
        //    << " Time (s): " << summary.total_time_in_seconds << "\n"
        //    << endl;
    }
}
#endif