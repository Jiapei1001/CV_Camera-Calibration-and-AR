#include "calibration.hpp"

#include <dirent.h>
#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
using namespace calibration;

// Finds the positions of internal corners of the chessboard.
std::vector<cv::Point2f> calibration::detectCorners(cv::Mat &src, cv::Size &boardSize) {
    // Reference: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
    // Sample usage of detecting and drawing chessboard corners
    std::vector<cv::Point2f> corner_set;

    bool cornersFound = cv::findChessboardCorners(src, boardSize, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

    // https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
    if (cornersFound) {
        Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        Size winSize(5, 5);
        Size zeroZone(-1, -1);
        cv::cornerSubPix(gray, corner_set, winSize, zeroZone, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

        // print out the cornet sets
        // printf("new corner set:\n");
        // for (int i = 0; i < boardSize.height; i++) {
        //     for (int j = 0; j < boardSize.width; j++) {
        //         int x = i * boardSize.height + j;
        //         Point2f p = corner_set[x];
        //         printf("[%.2f,  %.2f] ", p.x, p.y);
        //     }
        //     printf("\n");
        // }
    }

    // draw
    cv::drawChessboardCorners(src, boardSize, Mat(corner_set), cornersFound);

    return corner_set;
}

// Get the 3D world unit coordinates of the chessboard
std::vector<cv::Point3f> calibration::get3DWorldUnits(cv::Size &boardSize) {
    vector<cv::Point3f> point_set;

    // Note that if the (0, 0, 0) point is in the upper left corner, then the first point on the next row will be (0, -1, 0)
    // if the Z-axis comes towards the viewer.
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            // upper left is origin, Z-axis is 0
            point_set.push_back(Point3f(j, -i, 0));
        }
    }

    return point_set;
}

// print out information after camera calibration
void calibration::printCalibrateCameraInfo(cv::Mat &cameraMatrix, cv::Mat &distCoeffs, double error) {
    printf("\ncamera matrix:\n");
    for (int i = 0; i < cameraMatrix.rows; i++) {
        for (int j = 0; j < cameraMatrix.cols; j++) {
            printf("%.2lf ", cameraMatrix.at<double>(i, j));
        }
        printf("\n");
    }

    printf("\ndistortion coefficients: [ ");
    for (int i = 0; i < distCoeffs.rows; i++) {
        printf("%lf ", distCoeffs.at<double>(i, 0));
    }
    printf(" ]\n");

    printf("\nerror: %lf\n\n", error);
}
