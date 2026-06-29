/**
 * YOLO ONNX 推理实现骨架。
 * 完整 ONNX Runtime 集成需在目标平台安装 onnxruntime 开发包后编译。
 * 当前提供 OpenCV 预处理/后处理逻辑，便于对接 ONNX Session。
 */

#include "yolo_detector.h"

#include <algorithm>
#include <stdexcept>

YOLODetector::YOLODetector(const std::string& model_path, float conf_threshold)
    : session_(nullptr), conf_threshold_(conf_threshold), input_w_(640), input_h_(640) {
    // TODO: 初始化 Ort::Env, Ort::Session
    // 参考: https://onnxruntime.ai/docs/get-started/with-cpp.html
    (void)model_path;
    class_names_ = {"class0", "class1"};
}

YOLODetector::~YOLODetector() {
    // TODO: 释放 Ort Session
}

cv::Mat YOLODetector::preprocess(const cv::Mat& image) {
    cv::Mat resized, rgb, blob;
    cv::resize(image, resized, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(blob, CV_32F, 1.0 / 255.0);
    return blob;
}

std::vector<Detection> YOLODetector::postprocess(const cv::Mat& image, const float* output, int out_size) {
    (void)output;
    (void)out_size;
    // TODO: 解析 YOLOv8 ONNX 输出 tensor [1, 4+nc, 8400]
    return {};
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Empty image");
    }
    cv::Mat blob = preprocess(image);
    (void)blob;
    // TODO: Run ONNX inference
    return {};
}

cv::Mat YOLODetector::draw(const cv::Mat& image, const std::vector<Detection>& dets) {
    cv::Mat out = image.clone();
    for (const auto& d : dets) {
        cv::rectangle(out, d.box, cv::Scalar(0, 255, 0), 2);
        std::string label = d.label + " " + std::to_string(d.confidence).substr(0, 4);
        cv::putText(out, label, d.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    return out;
}
