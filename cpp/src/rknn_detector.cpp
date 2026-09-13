#include "rknn_detector.h"

#include <fstream>
#include <stdexcept>

namespace {
void check(int ret, const char* what) {
    if (ret != RKNN_SUCC) {
        throw std::runtime_error(std::string(what) + " failed, ret=" + std::to_string(ret));
    }
}
}  // namespace

RknnDetector::RknnDetector(const std::string& model_path) {
    std::ifstream f(model_path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("RKNN model not found: " + model_path);
    }
    std::vector<char> model(f.tellg());
    f.seekg(0);
    if (!f.read(model.data(), model.size())) {
        throw std::runtime_error("Failed to read RKNN model: " + model_path);
    }
    f.close();

    check(rknn_init(&ctx_, model.data(), model.size(), 0, nullptr), "rknn_init");

    rknn_input_output_num io{};
    check(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "rknn_query io num");
    n_out_ = io.n_output;

    in_attr_.index = 0;
    check(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_)), "rknn_query input attr");
    out_attr_.index = 0;
    check(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr_, sizeof(out_attr_)), "rknn_query output attr");
}

RknnDetector::~RknnDetector() {
    if (ctx_) {
        rknn_destroy(ctx_);
    }
}

std::vector<float> RknnDetector::infer(const cv::Mat& rgb) {
    uint32_t expect = 1;
    for (uint32_t i = 0; i < in_attr_.n_dims; ++i) {
        expect *= in_attr_.dims[i];
    }
    if (!rgb.isContinuous() || rgb.type() != CV_8UC3 || rgb.total() * 3 != expect) {
        throw std::runtime_error("infer: input must be a continuous NHWC uint8 RGB image matching model input");
    }

    // Model input is fp16; passing a uint8 buffer lets the runtime convert,
    // same path rknn-toolkit-lite2 takes with a uint8 numpy array.
    rknn_input in{};
    in.index = 0;
    in.buf = rgb.data;
    in.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    in.pass_through = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    check(rknn_inputs_set(ctx_, 1, &in), "rknn_inputs_set");
    check(rknn_run(ctx_, nullptr), "rknn_run");

    std::vector<rknn_output> outs(n_out_);
    for (uint32_t i = 0; i < n_out_; ++i) {
        outs[i].want_float = 1;
        outs[i].index = i;
    }
    check(rknn_outputs_get(ctx_, n_out_, outs.data(), nullptr), "rknn_outputs_get");

    const float* data = static_cast<const float*>(outs[0].buf);
    std::vector<float> result(data, data + outs[0].size / sizeof(float));
    rknn_outputs_release(ctx_, n_out_, outs.data());
    return result;
}
