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

    if (infile.is_open()) {
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
    } else {
        printf("calibration file cannot be opened. check its input path....\n");
        exit(-1);
    }
}