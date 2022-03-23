// calibration.hpp

#ifndef calibration_hpp
#define calibration_hpp

#include <opencv2/core/mat.hpp>

namespace calibration {
std::vector<cv::Point2f> detectCorners(cv::Mat &src, cv::Size &boardSize);
}  // namespace calibration

#endif /* calibration_hpp */