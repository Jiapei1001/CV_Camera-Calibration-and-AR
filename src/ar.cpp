#include "ar.hpp"

#include <dirent.h>
#include <math.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using namespace ar;

Scalar R = Scalar(0, 0, 255);
Scalar G = Scalar(0, 255, 0);
Scalar B = Scalar(255, 0, 0);

// Load camera calibration info from a .txt file
void ar::readCameraCalibrationInfo(const char *cameraCalibrationFile, cv::Mat &cameraMatrix, std::vector<double> &coeffs) {
    ifstream infile(cameraCalibrationFile);

    if (!infile.is_open()) {
        printf("calibration file cannot be opened. check its input path....\n");
        exit(-1);
    } else {
        string line;

        // load cameraMatrix info
        printf("\nloading cameraMatrix info...\n");
        for (int i = 0; i < 3; i++) {
            std::getline(infile, line);

            // split a string by delimeter
            // https://www.techiedelight.com/split-string-cpp-using-delimiter/
            size_t start;
            size_t end = 0;

            int idx = 0;
            while ((start = line.find_first_not_of(" ", end)) != std::string::npos) {
                end = line.find(" ", start);
                string temp = line.substr(start, end - start);

                cameraMatrix.at<double>(i, idx) = stod(temp);
                printf("%lf\n", cameraMatrix.at<double>(i, idx));
                idx++;
            }
        }

        printf("\nloading co-efficients info...\n");
        std::getline(infile, line);

        // https://www.techiedelight.com/split-string-cpp-using-delimiter/
        size_t start;
        size_t end = 0;

        int idx = 0;
        while ((start = line.find_first_not_of(" ", end)) != std::string::npos) {
            end = line.find(" ", start);
            string temp = line.substr(start, end - start);

            coeffs.push_back(stod(temp));
            printf("%lf\n", coeffs[idx]);
            idx++;
        }
    }
}

// Project 3D Axes:
// use the projectPoints function to project the 3D points corresponding to the four outside corners of the chessboard onto the image plane
// in real time as the chessboard or camera moves around.
void ar::project3DAxes(cv::Mat &frame, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::Mat &rvec, cv::Mat &tvec) {
    // projectPoints()
    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c

    std::vector<Point3f> axesPointsIn3DUnits{
        {0, 0, 0},   // 0
        {1, 0, 0},   // x
        {0, -1, 0},  // y
        {0, 0, 1}};  // z
    std::vector<Point2f> axesPointsInImage;

    cv::projectPoints(axesPointsIn3DUnits, rvec, tvec, cameraMatrix, distCoeffs, axesPointsInImage);

    // draw axes
    cv::line(frame, axesPointsInImage[0], axesPointsInImage[1], R, 4);  // x
    cv::line(frame, axesPointsInImage[0], axesPointsInImage[2], G, 4);  // y
    cv::line(frame, axesPointsInImage[0], axesPointsInImage[3], B, 4);  // z
}
