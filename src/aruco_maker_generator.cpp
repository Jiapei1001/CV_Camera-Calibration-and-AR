#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

// Entry function to generate Aruco Makers
int main(int argc, char *argv[]) {
    Mat markerImage1;
    Mat markerImage2;
    Mat markerImage3;
    Mat markerImage4;

    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // Reference - Maker creation
    // https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    // The first parameter is the Dictionary object previously created.
    // The second parameter is the marker id, in this case the marker 23 of the dictionary DICT_6X6_250.
    // The third parameter, 200, is the size of the output marker image.
    // The fourth parameter is the output image.
    // The last parameter is an optional parameter to specify the width of the marker black border.
    aruco::drawMarker(dictionary, 12, 200, markerImage1, 1);
    aruco::drawMarker(dictionary, 22, 200, markerImage2, 1);
    aruco::drawMarker(dictionary, 32, 200, markerImage3, 1);
    aruco::drawMarker(dictionary, 42, 200, markerImage4, 1);

    imwrite("marker1.png", markerImage1);
    imwrite("marker2.png", markerImage2);
    imwrite("marker3.png", markerImage3);
    imwrite("marker4.png", markerImage4);
}