#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "rknn_api.h"

// RAII wrapper around librknnrt, mirrors src/yolo11/rknn_infer.py.
class RknnDetector {
  public:
    explicit RknnDetector(const std::string& model_path);
    ~RknnDetector();

    RknnDetector(const RknnDetector&) = delete;
    RknnDetector& operator=(const RknnDetector&) = delete;

    // rgb: continuous NHWC uint8 image matching the model input WxH.
    // Returns output[0] converted to float32.
    std::vector<float> infer(const cv::Mat& rgb);

    const rknn_tensor_attr& input_attr() const { return in_attr_; }
    const rknn_tensor_attr& output_attr() const { return out_attr_; }

  private:
    rknn_context ctx_ = 0;
    uint32_t n_out_ = 0;
    rknn_tensor_attr in_attr_{};
    rknn_tensor_attr out_attr_{};
};
