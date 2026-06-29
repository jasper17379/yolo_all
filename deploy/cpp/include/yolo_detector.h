#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    std::string label;
};

class YOLODetector {
public:
    YOLODetector(const std::string& model_path, float conf_threshold = 0.25f);
    ~YOLODetector();

    std::vector<Detection> detect(const cv::Mat& image);
    cv::Mat draw(const cv::Mat& image, const std::vector<Detection>& dets);

private:
    void* session_;  // Ort::Session placeholder - requires onnxruntime headers at build time
    float conf_threshold_;
    std::vector<std::string> class_names_;
    int input_w_;
    int input_h_;

    cv::Mat preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(const cv::Mat& image, const float* output, int out_size);
};
