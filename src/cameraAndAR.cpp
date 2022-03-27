#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ar.hpp"
#include "calibration.hpp"

using namespace cv;
using namespace std;
using namespace ar;
using namespace calibration;

/* Helper method to check and print out loaded info.*/
void checkLoadedInfo(cv::Mat &cameraMatrix, cv::Mat &distCoeffs) {
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

/* Helper method to print out realtime rotation and translation result.*/
void printRealtimeResult(cv::Mat &rvec, cv::Mat &tvec) {
    // print out the rotation and translation data in real time
    printf("Rotation vectors: [");
    for (int i = 0; i < 3; i++) {
        printf("%lf", rvec.at<double>(0, i));
    }
    printf("]\n");
    printf("Translation vectors: [");
    for (int i = 0; i < 3; i++) {
        printf("%lf", tvec.at<double>(0, i));
    }
    printf("]\n");
}

/*
Helper method to starts a video loop.
For each frame, it tries to detect a chessboard.
If found, it grabs the locations of the corners, and then uses solvePNP to get the board's pose (rotation and translation).
*/
int loadVideo(cv::Mat &cameraMatrix, cv::Mat &distCoeffs) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);  // identifies a window

    cv::Mat frame;
    // must pass capdev to frame, to get updated frame size for initiating other Mat as below
    *capdev >> frame;

    Size boardSize(8, 6);

    int idx = 1;

    for (;;) {
        *capdev >> frame;  // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        char key = waitKey(10);
        if (key == 'q') {
            break;
        }

        std::vector<cv::Point3f> point_set;
        std::vector<Point2f> corner_set;
        // https://stackoverflow.com/questions/15245262/opencv-mat-element-types-and-their-sizes
        cv::Mat rvec = Mat::zeros(1, 3, DataType<double>::type);
        cv::Mat tvec = Mat::zeros(1, 3, DataType<double>::type);

        // load 3D world units
        point_set = calibration::get3DWorldUnits(boardSize);

        bool foundChessBoard = cv::findChessboardCorners(frame, boardSize, corner_set);
        if (foundChessBoard) {
            // Finds an object pose from 3D-2D point correspondences.
            // This function returns the rotation and the translation vectors that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame.
            // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
            cv::solvePnP(point_set, corner_set, cameraMatrix, distCoeffs, rvec, tvec);

            // printRealtimeResult(rvec, tvec);
            ar::project3DAxes(frame, cameraMatrix, distCoeffs, rvec, tvec);

            ar::project3DTriangular(frame, 4, -1, cameraMatrix, distCoeffs, rvec, tvec);

            // save the frame as an image
            if (key == 'w') {
                string fname = "../data/ar/image_" + to_string(idx) + ".jpg";
                imwrite(fname, frame);
                idx++;
            }
        }

        imshow("Video", frame);
    }

    delete capdev;
    return (0);
}

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

    checkLoadedInfo(cameraMatrix, distCoeffs);

    loadVideo(cameraMatrix, distCoeffs);
}