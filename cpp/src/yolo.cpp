#include "yolo.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

cv::Mat yolo::letterbox_rgb(const cv::Mat& bgr, int size, LetterboxMeta& meta) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || size <= 0) {
        throw std::invalid_argument("letterbox_rgb: expect HWC BGR uint8 image and size > 0");
    }
    float scale = std::min(size / static_cast<float>(bgr.cols), size / static_cast<float>(bgr.rows));
    int rw = static_cast<int>(std::lround(bgr.cols * scale));
    int rh = static_cast<int>(std::lround(bgr.rows * scale));

    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    int pad_x = (size - rw) / 2, pad_y = (size - rh) / 2;
    cv::Mat resized;
    cv::resize(bgr, resized, {rw, rh}, 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, rw, rh)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    meta = {bgr.cols, bgr.rows, size, scale, pad_x, pad_y};
    return canvas;
}

namespace {

float iou(const yolo::Detection& a, const yolo::Detection& b) {
    float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float inter = iw * ih;
    float uni = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1) +
                std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// Class-aware greedy NMS, score descending.
void nms(std::vector<yolo::Detection>& dets, float thresh) {
    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<char> suppressed(dets.size(), 0);
    std::vector<yolo::Detection> kept;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[j].class_id == dets[i].class_id && iou(dets[j], dets[i]) > thresh) {
                suppressed[j] = 1;
            }
        }
    }
    dets = std::move(kept);
}

}  // namespace

std::vector<yolo::Detection> yolo::postprocess(const float* out, const uint32_t* dims, int n_dims,
                                             const LetterboxMeta& meta, float conf, float nms_thresh) {
    if (n_dims < 2) {
        throw std::runtime_error("postprocess: unsupported YOLO output rank");
    }
    // Last two dims are (rows, cols); squeeze leading batch dims like the Python version.
    uint32_t r = dims[n_dims - 2], c = dims[n_dims - 1];
    bool chan_first = (r >= 5 && r < c);  // [C,N] layout
    uint32_t n_anchor = chan_first ? c : r;
    uint32_t n_field = chan_first ? r : c;
    if (n_field < 5) {
        throw std::runtime_error("postprocess: unsupported YOLO output shape");
    }
    int n_cls = static_cast<int>(n_field) - 4;
    auto at = [&](int f, uint32_t a) -> float {
        return chan_first ? out[f * n_anchor + a] : out[a * n_field + f];
    };

    // Ultralytics int8 exports normalize box coords to [0,1]; rescale when detected.
    float box_max = 0.f;
    for (uint32_t a = 0; a < n_anchor; ++a) {
        for (int f = 0; f < 4; ++f) box_max = std::max(box_max, at(f, a));
    }
    float coord_scale = box_max <= 2.f ? static_cast<float>(meta.input_size) : 1.f;

    float ow = static_cast<float>(meta.orig_w), oh = static_cast<float>(meta.orig_h);
    std::vector<Detection> dets;
    for (uint32_t a = 0; a < n_anchor; ++a) {
        int best = 0;
        float bs = at(4, a);
        for (int k = 1; k < n_cls; ++k) {
            float s = at(4 + k, a);
            if (s > bs) { bs = s; best = k; }
        }
        if (bs < conf) continue;

        float x = at(0, a) * coord_scale, y = at(1, a) * coord_scale;
        float w = at(2, a) * coord_scale, h = at(3, a) * coord_scale;
        Detection d;
        d.class_id = best;
        d.score = bs;
        d.x1 = std::clamp(((x - w / 2.f) - meta.pad_x) / meta.scale, 0.f, ow);
        d.y1 = std::clamp(((y - h / 2.f) - meta.pad_y) / meta.scale, 0.f, oh);
        d.x2 = std::clamp(((x + w / 2.f) - meta.pad_x) / meta.scale, 0.f, ow);
        d.y2 = std::clamp(((y + h / 2.f) - meta.pad_y) / meta.scale, 0.f, oh);
        dets.push_back(d);
    }
    nms(dets, nms_thresh);
    return dets;
}

float yolo::iob(const Detection& inner, const Detection& outer) {
    float iw = std::max(0.f, std::min(inner.x2, outer.x2) - std::max(inner.x1, outer.x1));
    float ih = std::max(0.f, std::min(inner.y2, outer.y2) - std::max(inner.y1, outer.y1));
    float area = std::max(0.f, inner.x2 - inner.x1) * std::max(0.f, inner.y2 - inner.y1);
    return area > 0.f ? iw * ih / area : 0.f;
}
