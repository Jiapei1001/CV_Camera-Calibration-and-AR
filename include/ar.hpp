// ar.hpp

#ifndef ar_hpp
#define ar_hpp

#include <opencv2/core/mat.hpp>

const cv::Scalar R = cv::Scalar(0, 0, 255);
const cv::Scalar G = cv::Scalar(0, 255, 0);
const cv::Scalar B = cv::Scalar(255, 0, 0);
const cv::Scalar ORIANGE = cv::Scalar(0, 140, 255);
const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
const cv::Scalar GRAY = cv::Scalar(200, 200, 200);

namespace ar {
void readCameraCalibrationInfo(const char *cameraCalibrationFile, cv::Mat &cameraMatrix, std::vector<double> &coeffs);
void project3DAxes(cv::Mat &frame, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::Mat &rvec, cv::Mat &tvec);
void project3DTriangular(cv::Mat &frame, float x, float y, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::Mat &rvec, cv::Mat &tvec);
}  // namespace ar

#endif /* ar_hpp */