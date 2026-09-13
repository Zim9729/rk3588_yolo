#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rknn_detector.h"
#include "yolo.h"

namespace fs = std::filesystem;

namespace {

const std::map<std::string, std::string> PART_NAMES = {
    {"tb", "碳滑板"}, {"yj", "羊角"}, {"yw", "异物"}};
// 与 JC24 toml 中 stage2_overlap_threshold 一致:yw 框落入部件区域面积占比阈值
constexpr float OVERLAP_THRESHOLD = 0.5f;

struct Args {
    std::string image = "3C_Test.jpg";
    std::string stage1_model = "models_arm/3C_stage1_fp16.rknn";
    std::string stage2_model = "models_arm/3C_stage2_fp16.rknn";
    std::string stage2_config = "configs/stage2.yaml";
    std::string output = "outputs/two_stage_result.jpg";
    float conf = -1.f;
    float overlap = OVERLAP_THRESHOLD;
};

[[noreturn]] void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " [--image P] [--stage1-model P] [--stage2-model P] [--stage2-config P]\n"
                 "            [--output P] [--conf F] [--overlap F]\n";
    std::exit(1);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i], key, val;
        if (s.rfind("--", 0) != 0) usage(argv[0]);
        auto eq = s.find('=');
        if (eq != std::string::npos) {
            key = s.substr(2, eq - 2);
            val = s.substr(eq + 1);
        } else {
            key = s.substr(2);
            if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) usage(argv[0]);
            val = argv[++i];
        }
        if (key == "image") a.image = val;
        else if (key == "stage1-model") a.stage1_model = val;
        else if (key == "stage2-model") a.stage2_model = val;
        else if (key == "stage2-config") a.stage2_config = val;
        else if (key == "output") a.output = val;
        else if (key == "conf") a.conf = std::stof(val);
        else if (key == "overlap") a.overlap = std::stof(val);
        else usage(argv[0]);
    }
    return a;
}

std::string trim(std::string s) {
    auto b = s.find_first_not_of(" \t\r\n");
    auto e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

// Flat "key: value" YAML subset; enough for configs/*.yaml.
std::unordered_map<std::string, std::string> load_flat_yaml(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Config file not found: " + path.string());
    std::unordered_map<std::string, std::string> m;
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string k = trim(line.substr(0, colon)), v = trim(line.substr(colon + 1));
        if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front()) {
            v = v.substr(1, v.size() - 2);
        }
        if (!k.empty()) m[k] = v;
    }
    return m;
}

struct Config {
    int img_size;
    fs::path labels_path;
    float conf;
    float nms;
};

Config load_config(const fs::path& path) {
    auto m = load_flat_yaml(path);
    const char* required[] = {"img_size", "labels", "conf_threshold", "nms_threshold", "input_format"};
    std::string missing;
    for (auto k : required)
        if (!m.count(k)) missing += (missing.empty() ? "" : ", ") + std::string(k);
    if (!missing.empty()) throw std::runtime_error("Missing config keys in " + path.string() + ": " + missing);
    if (m["input_format"] != "nhwc") throw std::runtime_error("Only nhwc input_format is supported");

    Config c{std::stoi(m["img_size"]), {}, std::stof(m["conf_threshold"]), std::stof(m["nms_threshold"])};
    fs::path lp = m["labels"];
    c.labels_path = lp.is_absolute() ? lp : fs::absolute(path.parent_path() / lp).lexically_normal();
    if (c.img_size <= 0) throw std::runtime_error("img_size must be positive");
    if (c.conf < 0.f || c.conf > 1.f || c.nms < 0.f || c.nms > 1.f) {
        throw std::runtime_error("conf_threshold and nms_threshold must be between 0 and 1");
    }
    return c;
}

std::vector<std::string> load_labels(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Labels file not found: " + path.string());
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (!line.empty()) labels.push_back(line);
    }
    if (labels.empty()) throw std::runtime_error("No labels found in: " + path.string());
    return labels;
}

std::vector<yolo::Detection> run_stage(RknnDetector& det, const cv::Mat& img, int img_size,
                                       float conf, float nms) {
    yolo::LetterboxMeta meta;
    cv::Mat tensor = yolo::letterbox_rgb(img, img_size, meta);
    std::vector<float> out = det.infer(tensor);
    const rknn_tensor_attr& attr = det.output_attr();
    return yolo::postprocess(out.data(), attr.dims, static_cast<int>(attr.n_dims), meta, conf, nms);
}

std::string part_cn(const std::string& name) {
    auto it = PART_NAMES.find(name);
    return it == PART_NAMES.end() ? name : it->second;
}

std::string fmt2(float v) {
    char b[16];
    std::snprintf(b, sizeof b, "%.2f", v);
    return b;
}

struct DrawItem {
    float x1, y1, x2, y2;
    std::string text;
    cv::Scalar color;
};

}  // namespace

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);
    if (!fs::exists(args.image)) throw std::runtime_error("Image not found: " + args.image);

    Config cfg = load_config(args.stage2_config);
    std::vector<std::string> stage2_labels = load_labels(cfg.labels_path);
    float conf = args.conf < 0.f ? cfg.conf : args.conf;

    cv::Mat image = cv::imread(args.image, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Failed to read image: " + args.image);
    int img_w = image.cols, img_h = image.rows;
    cv::Mat canvas = image.clone();
    std::vector<DrawItem> results;

    {
        RknnDetector det1(args.stage1_model), det2(args.stage2_model);

        auto stage1_dets = run_stage(det1, image, cfg.img_size, conf, cfg.nms);
        std::printf("[stage1] %zu Pantograph_Area:\n", stage1_dets.size());
        for (const auto& d : stage1_dets) {
            std::printf("  Pantograph_Area conf=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
                        d.score, d.x1, d.y1, d.x2, d.y2);
            results.push_back({d.x1, d.y1, d.x2, d.y2,
                               "Pantograph_Area " + fmt2(d.score),
                               {0, 200, 0}});
        }

        for (size_t ai = 0; ai < stage1_dets.size(); ++ai) {
            int ax1 = std::max(0, static_cast<int>(std::lround(stage1_dets[ai].x1)));
            int ay1 = std::max(0, static_cast<int>(std::lround(stage1_dets[ai].y1)));
            int ax2 = std::min(img_w, static_cast<int>(std::lround(stage1_dets[ai].x2)));
            int ay2 = std::min(img_h, static_cast<int>(std::lround(stage1_dets[ai].y2)));
            if (ax2 <= ax1 || ay2 <= ay1) continue;
            cv::Mat crop = image(cv::Rect(ax1, ay1, ax2 - ax1, ay2 - ay1));

            auto stage2_dets = run_stage(det2, crop, cfg.img_size, conf, cfg.nms);
            std::vector<yolo::Detection> parts, foreigns;
            for (auto& d : stage2_dets) {
                // 映射回原图坐标
                d.x1 += ax1; d.y1 += ay1; d.x2 += ax1; d.y2 += ay1;
                if (d.class_id < 0 || d.class_id >= static_cast<int>(stage2_labels.size()))
                    throw std::runtime_error("stage2 class_id out of labels range");
                (stage2_labels[d.class_id] == "yw" ? foreigns : parts).push_back(d);
            }

            std::printf("[stage2] area#%zu: %zu parts, %zu yw raw\n", ai, parts.size(), foreigns.size());
            for (const auto& d : parts) {
                const std::string& name = stage2_labels[d.class_id];
                std::printf("  %s(%s) conf=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
                            name.c_str(), part_cn(name).c_str(), d.score, d.x1, d.y1, d.x2, d.y2);
                results.push_back({d.x1, d.y1, d.x2, d.y2,
                                   name + " " + fmt2(d.score), {255, 128, 0}});
            }

            for (const auto& d : foreigns) {
                const yolo::Detection* best_part = nullptr;
                float best_iob = 0.f;
                for (const auto& p : parts) {
                    float ov = yolo::iob(d, p);
                    if (ov > best_iob) { best_iob = ov; best_part = &p; }
                }
                if (!best_part || best_iob < args.overlap) continue;  // 不落在任何部件上的异物不输出
                const std::string& pname = stage2_labels[best_part->class_id];
                std::printf("  %s异物 (yw->%s, iob=%.2f) conf=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
                            part_cn(pname).c_str(), pname.c_str(), best_iob, d.score,
                            d.x1, d.y1, d.x2, d.y2);
                results.push_back({d.x1, d.y1, d.x2, d.y2,
                                   pname + "_yw " + fmt2(d.score), {0, 0, 255}});
            }
        }
    }

    for (const auto& r : results) {
        int x1 = static_cast<int>(std::lround(r.x1)), y1 = static_cast<int>(std::lround(r.y1));
        int x2 = static_cast<int>(std::lround(r.x2)), y2 = static_cast<int>(std::lround(r.y2));
        cv::rectangle(canvas, {x1, y1}, {x2, y2}, r.color, 2);
        cv::putText(canvas, r.text, {x1, std::max(y1 - 4, 12)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, r.color, 2, cv::LINE_AA);
    }

    fs::path out_path(args.output);
    if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path());
    if (!cv::imwrite(args.output, canvas)) {
        throw std::runtime_error("Failed to write output image: " + args.output);
    }
    std::printf("Output: %s\n", args.output.c_str());
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
