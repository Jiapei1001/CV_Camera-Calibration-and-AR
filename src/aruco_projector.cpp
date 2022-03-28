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
    std::cout << "Detect markers and show their 3D axises \t -key 'd'" << std::endl;
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

// Map a source image to the markers' area in the video frame
void mapImageToMaker(cv::Mat &cameraMatrix, std::vector<double> &distCoeffs) {
    cv::VideoCapture videoCap;
    cv::VideoWriter videoWriter;

    videoCap.open(0);
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::Mat imgSrc = cv::imread("../data/image_source.jpg");
    // cv::imshow("image", imgSrc);

    Mat concatenatedOutput;
    Mat frame;

    while (videoCap.grab()) {
        cv::Mat image, imageCopy;
        cv::Mat mappedResult;
        videoCap.retrieve(image);
        image.copyTo(imageCopy);

        videoCap >> frame;

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners, failedCandidates;

        // Initialize the detector parameters using default values
        Ptr<DetectorParameters> parameters = DetectorParameters::create();

        // detect markers
        // corner index
        // markerCorners is the list of corners of the detected markers. For each marker, its four corners are returned in their original order (which is clockwise starting with top left).
        // So, the first corner is the top left corner, followed by the top right, bottom right and bottom left.
        // https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
        cv::aruco::detectMarkers(frame, dictionary, corners, ids, parameters, failedCandidates);

        // if at least one marker detected
        if (ids.size() > 0) {
            // locate the points in the destination frame
            vector<Point> pts_dst;
            float scalingFactor = 0.015;  // 0.015;

            Point pt1, pt2, pt3, pt4;

            // top left
            std::vector<int>::iterator it = std::find(ids.begin(), ids.end(), 12);
            int index = std::distance(ids.begin(), it);
            // top left marker's top right corner
            pt1 = corners.at(index).at(1);

            // top right
            it = std::find(ids.begin(), ids.end(), 22);
            index = std::distance(ids.begin(), it);
            // top right marker's bottom right corner
            pt2 = corners.at(index).at(2);

            float distance = norm(pt1 - pt2);
            pts_dst.push_back(Point(pt1.x - round(scalingFactor * distance), pt1.y - round(scalingFactor * distance)));
            pts_dst.push_back(Point(pt2.x + round(scalingFactor * distance), pt2.y - round(scalingFactor * distance)));

            // bottom right
            it = std::find(ids.begin(), ids.end(), 32);
            index = std::distance(ids.begin(), it);
            // bottom right marker's top left corner
            pt3 = corners.at(index).at(0);
            pts_dst.push_back(Point(pt3.x + round(scalingFactor * distance), pt3.y + round(scalingFactor * distance)));

            // bottom left
            it = std::find(ids.begin(), ids.end(), 32);
            index = std::distance(ids.begin(), it);
            // bottom left marker's top left corner
            pt4 = corners.at(index).at(0);
            pts_dst.push_back(Point(pt4.x - round(scalingFactor * distance), pt4.y - round(scalingFactor * distance)));

            // corner points of the new source image
            vector<Point> pts_src;
            // top left
            pts_src.push_back(Point(0, 0));
            // top right
            pts_src.push_back(Point(imgSrc.cols, 0));
            // bottom right
            pts_src.push_back(Point(imgSrc.cols, imgSrc.rows));
            // bottom left
            pts_src.push_back(Point(0, imgSrc.rows));

            // calculate homography
            // A Homography is a transformation ( a 3×3 matrix ) that maps the points in one image to the corresponding points in the other image.
            // Reference - https://learnopencv.com/homography-examples-using-opencv-python-c/
            cv::Mat homo = cv::findHomography(pts_src, pts_dst);

            // Map the source image to the mapped image using the homography
            cv::Mat mappedImage;
            warpPerspective(imgSrc, mappedImage, homo, frame.size(), INTER_CUBIC);

            // Mask as the region to copy from the mapped image into the original frame
            cv::Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
            fillConvexPoly(mask, pts_dst, Scalar(255, 255, 255), LINE_AA);

            // Erode the mask to not copy the boundary effects from the mapping process
            cv::Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
            erode(mask, mask, element);

            // Map the new source image into the mask area
            mappedResult = frame.clone();
            mappedImage.copyTo(mappedResult, mask);

            // cv::aruco::drawDetectedMarkers(mappedResult, corners, ids);

            hconcat(frame, mappedResult, concatenatedOutput);
        }

        cv::imshow("out", frame);
        // videoWriter.write(mappedResult);

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

    if (strcmp(argv[2], "d") == 0) {
        detectAndShowMarkers(cameraMatrix, coeffs);
    } else if (strcmp(argv[2], "m") == 0) {
        mapImageToMaker(cameraMatrix, coeffs);
    } else {
        cout << "The specified mode is not correct.\n";
        exit(-1);
    }

    // NOTE: must add waitKey, or the program will terminate, without showing the result images
    waitKey(0);
    printf("Terminating\n");

    return 0;
}