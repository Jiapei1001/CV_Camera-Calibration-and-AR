// ar.hpp

#ifndef ar_hpp
#define ar_hpp

#include <opencv2/core/mat.hpp>

namespace ar {
void readCameraCalibrationInfo(const char *cameraCalibrationFile, cv::Mat &cameraMatrix, std::vector<double> &coeffs);
void project3DAxes(cv::Mat &frame, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, cv::Mat &rvec, cv::Mat &tvec);
}  // namespace ar

#endif /* ar_hpp */