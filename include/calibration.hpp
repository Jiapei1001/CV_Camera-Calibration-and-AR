// calibration.hpp

#ifndef calibration_hpp
#define calibration_hpp

#include <opencv2/core/mat.hpp>

namespace calibration {
std::vector<cv::Point2f> detectCorners(cv::Mat &src, cv::Size &boardSize);
std::vector<cv::Point3f> get3DWorldUnits(cv::Size &boardSize);
void printCalibrateCameraInfo(cv::Mat &cameraMatrix, cv::Mat &distCoeffs, double error);
}  // namespace calibration

#endif /* calibration_hpp */