// ar.hpp

#ifndef ar_hpp
#define ar_hpp

#include <opencv2/core/mat.hpp>

namespace ar {
void readCameraCalibrationInfo(const char *cameraCalibrationFile, cv::Mat &cameraMatrix, std::vector<double> &coeffs);

}  // namespace ar

#endif /* ar_hpp */