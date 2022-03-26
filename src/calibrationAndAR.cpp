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

        imshow("Video", frame);
    }

    delete capdev;
    return (0);
}