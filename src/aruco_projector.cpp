#include <fstream>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "ar.hpp"

using namespace cv;
using namespace aruco;
using namespace std;
using namespace ar;

// Print out cmd options
void printOptions() {
    // Menu buttons:
    std::cout << "\nKeys for aruco projector:" << std::endl;
    std::cout << "Detect markers and show their 3D axises \t -key 'm'" << std::endl;
    std::cout << "Quit  \t\t\t\t\t\t -key 'q'\n"
              << std::endl;
}

// Detect aruco makers, and show their borders in the video frame
void detectAndShowMarkers(cv::Mat &cameraMatrix, std::vector<double> &distCoeffs) {
    cv::VideoCapture videoCap;
    videoCap.open(0);
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    while (videoCap.grab()) {
        cv::Mat image, imageCopy;
        videoCap.retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        // if at least one marker detected
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
            // draw axis for each marker
            for (int i = 0; i < ids.size(); i++) {
                cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
            }
        }
        cv::imshow("out", imageCopy);

        char key = (char)cv::waitKey(10);
        if (key == 'q' || key == 27) {
            break;
        }
    }
}

// Entry function to project a new image to the targeted area in the video frame,
// leveraging the aruco AR library.
// Reference - https://docs.opencv.org/3.4/d5/dae/tutorial_aruco_detection.html
int main(int argc, char *argv[]) {
    printOptions();

    // check for sufficient arguments
    if (argc < 3) {
        cout << "Please specify a file path to camera calibration file as #1 argument.\n";
        cout << "Please specify a mode as #2 argument.\n";
        exit(-1);
    }

    char cameraCalibrationFile[256];
    cv::Mat cameraMatrix(3, 3, CV_64FC1);
    std::strcpy(cameraCalibrationFile, argv[1]);

    std::vector<double> coeffs;
    ar::readCameraCalibrationInfo(cameraCalibrationFile, cameraMatrix, coeffs);

    if (strcmp(argv[2], "m") == 0) {
        detectAndShowMarkers(cameraMatrix, coeffs);
    } else {
        cout << "The specified mode is not correct.\n";
        exit(-1);
    }

    // NOTE: must add waitKey, or the program will terminate, without showing the result images
    waitKey(0);
    printf("Terminating\n");

    return 0;
}