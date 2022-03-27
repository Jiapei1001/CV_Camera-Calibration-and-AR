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
    cv::arrowedLine(frame, axesPointsInImage[0], axesPointsInImage[1], R, 2);  // x
    cv::arrowedLine(frame, axesPointsInImage[0], axesPointsInImage[2], G, 2);  // y
    cv::arrowedLine(frame, axesPointsInImage[0], axesPointsInImage[3], B, 2);  // z
}

// Project 3D Triangular:
// Reference - https://gist.github.com/MareArts/54011c365ec0d66d59562945df13dbfe
void ar::project3DTriangular(cv::Mat &frame, float x, float y, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<Point3f> axesPointsIn3DUnits{
        {x, y, 0},                 // upper left
        {x + 2.0f, y, 0},          // upper right
        {x, y - 2.0f, 0},          // bottom left
        {x + 2.0f, y - 2.0f, 0},   // bottom right
        {x, y, 4},                 // upper left
        {x + 2.0f, y, 4},          // upper right
        {x, y - 2.0f, 4},          // bottom left
        {x + 2.0f, y - 2.0f, 4},   // bottom right
        {x + 1.0f, y - 1.0f, 4}};  // center z
    std::vector<Point2f> axesPointsInImage;

    cv::projectPoints(axesPointsIn3DUnits, rvec, tvec, cameraMatrix, distCoeffs, axesPointsInImage);

    // draw
    // cv::rectangle(frame, axesPointsInImage[0], axesPointsInImage[3], YELLOW, cv::FILLED);
    cv::line(frame, axesPointsInImage[0], axesPointsInImage[1], R, 2);
    cv::line(frame, axesPointsInImage[1], axesPointsInImage[2], G, 2);
    cv::line(frame, axesPointsInImage[2], axesPointsInImage[3], B, 2);
    cv::line(frame, axesPointsInImage[3], axesPointsInImage[0], ORIANGE, 2);

    // top
    cv::line(frame, axesPointsInImage[4], axesPointsInImage[5], R, 2);
    cv::line(frame, axesPointsInImage[5], axesPointsInImage[6], G, 2);
    cv::line(frame, axesPointsInImage[6], axesPointsInImage[7], B, 2);
    cv::line(frame, axesPointsInImage[7], axesPointsInImage[4], ORIANGE, 2);

    // surround
    cv::line(frame, axesPointsInImage[0], axesPointsInImage[4], R, 2);
    cv::line(frame, axesPointsInImage[1], axesPointsInImage[5], G, 2);
    cv::line(frame, axesPointsInImage[2], axesPointsInImage[6], B, 2);
    cv::line(frame, axesPointsInImage[3], axesPointsInImage[7], ORIANGE, 2);

    // push points in 2D image into contours
    std::vector<Point2f> contours;
    contours.push_back(axesPointsInImage[0]);
    contours.push_back(axesPointsInImage[1]);
    contours.push_back(axesPointsInImage[4]);

    // draw filled surface
    // https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
    // cv::drawContours(frame, contours, 0, ORIANGE, 2);
}
