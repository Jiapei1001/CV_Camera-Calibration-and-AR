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
    }
    cv::drawChessboardCorners(src, boardSize, Mat(corner_set), cornersFound);
    return corner_set;
}
