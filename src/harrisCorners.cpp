#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

const cv::Scalar ORANGE = cv::Scalar(0, 140, 255);

/* Helper method to call harrisCorners method in openCV to detect and draw key points */
// Reference - https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
void detectAndDrawHarrisCorners(cv::Mat &frame) {
    // change to gray
    cv::Mat gray;
    cv::cvtColor(frame, gray, COLOR_BGR2GRAY);

    // output
    cv::Mat dst = Mat::zeros(frame.size(), CV_32FC1);  // 32-bit float

    // blockSize	Neighborhood size (see the details on cornerEigenValsAndVecs ).
    int blockSize = 2;
    // ksize	    Aperture parameter for the Sobel operator.
    int ksize = 3;
    // k	        Harris detector free parameter.
    double k = 0.04;

    cv::cornerHarris(gray, dst, blockSize, ksize, k);

    // threshold
    // https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    int threshold = 202;
    int max_threshold = 255;

    cv::Mat normed;
    cv::normalize(dst, normed, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // draw the key points as circles
    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            if ((int)normed.at<float>(i, j) > threshold) {
                cv::circle(frame, Point(j, i), 4, ORANGE, 2, 8, 0);
            }
        }
    }
}

/* Entry function to detect and draw harris corners for video frames */
int videoMode() {
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

        char key = waitKey(10);
        if (key == 'q') {
            break;
        }

        detectAndDrawHarrisCorners(frame);

        imshow("Video", frame);
    }

    delete capdev;
    return (0);
}

/* Entry function to detect and draw harris corners for an image */
int imageMode(char *imageFile) {
    if (strstr(imageFile, ".jpg") ||
        strstr(imageFile, ".png") ||
        strstr(imageFile, ".ppm") ||
        strstr(imageFile, ".tif")) {
        cv::Mat image;
        image = imread(string(imageFile));

        // check if new Mat is built
        if (image.data == NULL) {
            cout << "This new image" << imageFile << "cannot be loaded into cv::Mat\n";
            exit(-1);
        }
        detectAndDrawHarrisCorners(image);

        imshow("Image", image);
    } else {
        cout << "This new image" << imageFile << "cannot be loaded.\n";
    }

    return 0;
}

/*
  Entry function to detect and draw harris corners
  Reference: harrisCorners With OpenCV
  https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345
 */
int main(int argc, char *argv[]) {
    char imageFile[256];

    if (argc == 1) {
        videoMode();
    } else if (argc == 2) {
        strcpy(imageFile, argv[1]);
        imageMode(imageFile);
    } else {
        printf("Input arguments are not correct.\n");
        exit(-1);
    }

    // NOTE: must add waitKey, or the program will terminate, without showing the result images
    waitKey(0);
    printf("Terminating\n");

    return 0;
}