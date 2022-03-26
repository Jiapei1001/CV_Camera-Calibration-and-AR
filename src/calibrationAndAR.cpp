#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>

#include "calibration.hpp"

using namespace cv;
using namespace std;
using namespace calibration;

/*
  Entry function to the calibration
  Reference: Camera calibration With OpenCV
  https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
 */
int main(int argc, char *argv[]) {
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

    std::vector<cv::Point3f> point_set;
    std::vector<std::vector<cv::Point3f> > point_list;
    std::vector<std::vector<cv::Point2f> > corner_list;
    int idx = 0;

    // initialize camera matrix
    double camera_matrix[3][3] = {
        {1, 0, frame.cols / 2.0},
        {0, 1, frame.rows / 2.0},
        {0, 0, 1}};
    // Make the camera_matrix a 3x3 cv::Mat of type CV_64FC1
    cv::Mat cameraMatrix(3, 3, CV_64FC1, camera_matrix);
    cv::Mat distCoeffs = Mat::zeros(8, 1, CV_64F);

    // Output vector of rotation vectors (Rodrigues ) estimated for each pattern view (e.g. std::vector<cv::Mat>>).
    // That is, each i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter description) brings the calibration pattern from the object coordinate space (in which object points are specified) to the camera coordinate space.
    // In more technical terms, the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space to camera coordinate space.
    // Due to its duality, this tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate space.
    std::vector<cv::Mat> rvecs;

    // Output vector of translation vectors estimated for each pattern view, see parameter describtion above.
    std::vector<cv::Mat> tvecs;

    for (;;) {
        *capdev >> frame;  // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        std::vector<Point2f> corner_set = calibration::detectCorners(frame, boardSize);

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);

        // break the loop
        if (key == 'q') {
            break;
        }
        // make sure the frame is calibrated
        else if (key == 's' && corner_set.size() > 0) {
            // save the corner locations
            corner_list.push_back(corner_set);

            // create a point_set that specifies the 3D units of the corners in world coordinates
            point_set = calibration::get3DWorldUnits(boardSize);
            point_list.push_back(point_set);

            // save the frame as an image
            string fname = "../data/calibration/image_" + to_string(idx) + ".jpg";
            imwrite(fname, frame);
            idx++;
        }
        // calibrate the camera
        else if (key == 'c') {
            if (corner_list.size() < 5) {
                printf("save at least 5 calibration frames\n");
            } else {
                // cv::calibrateCamera
                // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
                double error = cv::calibrateCamera(point_list, corner_list, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

                // > half-pixel
                if (error > 0.5) {
                    printf("the error should be less than a half-pixel. please reran the calibration images.\n");
                    break;
                }

                // Print out the camera matrix and distortion coefficients after the calibration, along with the final re-projection error.
                calibration::printCalibrateCameraInfo(cameraMatrix, distCoeffs, error);
            }
        }

        imshow("Video", frame);
    }

    delete capdev;
    return (0);
}