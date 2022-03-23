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

    for (;;) {
        *capdev >> frame;  // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);

        // break the loop
        if (key == 'q') {
            break;
        }
    }

    delete capdev;
    return (0);
}