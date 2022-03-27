#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ar.hpp"

using namespace cv;
using namespace std;
using namespace ar;

/*
  Entry function to the AR
  Reference: solvePNP With OpenCV
  https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
 */
int main(int argc, char *argv[]) {
    char cameraCalibrationFile[256];
    // image of type CV_64FC1 is simple grayscale image and has only 1 channel:
    // image of type CV_64FC3 is colored image with 3 channels
    // CV_64F is the same as CV_64FC1
    // https://stackoverflow.com/questions/19248926/difference-of-opencv-mat-types
    cv::Mat cameraMatrix(3, 3, CV_64FC1);

    if (argc < 2) {
        cout << "Please give a file path to camera calibration file\n";
        exit(-1);
    }

    std::strcpy(cameraCalibrationFile, argv[1]);
    std::vector<double> coeffs;
    ar::readCameraCalibrationInfo(cameraCalibrationFile, cameraMatrix, coeffs);

    // check at least 5 images' info are loaded
    if (coeffs.size() < 5) {
        printf("\nCamera calibration info cannot be loaded. Please give a correct file path.\n");
        exit(-1);
    } else {
        printf("\nCamera calibration info has been loaded.\n");
    }

    // load coeffs to distCoeffs, no pre-defined distCoeffs' size
    cv::Mat distCoeffs = cv::Mat::zeros(coeffs.size(), 1, CV_64F);
    for (int i = 0; i < coeffs.size(); i++) {
        distCoeffs.at<double>(i, 0) = coeffs[i];
    }

    // check info are loaded correctly
    printf("\nCamera matrix: \n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%.2lf  ", cameraMatrix.at<double>(i, j));
        }
        printf("\n");
    }
    printf("\nDistortion ceofficients: ");
    for (int i = 0; i < distCoeffs.rows; i++) {
        printf("%lf ", distCoeffs.at<double>(i, 0));
    }
    printf("\n\n");

    
}