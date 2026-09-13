#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

namespace yolo {

struct Detection {
    int class_id;
    float score;
    float x1, y1, x2, y2;
};

struct LetterboxMeta {
    int orig_w = 0, orig_h = 0;
    int input_size = 0;
    float scale = 1.f;
    int pad_x = 0, pad_y = 0;
};

// BGR image -> square RGB uint8 tensor (NHWC), letterboxed to `size` x `size`.
cv::Mat letterbox_rgb(const cv::Mat& bgr, int size, LetterboxMeta& meta);

// Decode a YOLO11 output tensor into detections in original-image coordinates.
// `dims`/`n_dims` describe the raw output, e.g. {1, 7, 34000} or {1, 34000, 7}.
std::vector<Detection> postprocess(const float* out, const uint32_t* dims, int n_dims,
                                   const LetterboxMeta& meta, float conf, float nms_thresh);

// Intersection over the inner box area (foreign object attribution).
float iob(const Detection& inner, const Detection& outer);

}  // namespace yolo
