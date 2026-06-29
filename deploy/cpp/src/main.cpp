#include "yolo_detector.h"

#include <iostream>

int main(int argc, char* argv[]) {
    std::string model = "model.onnx";
    std::string source = "test.jpg";
    float conf = 0.25f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) conf = std::stof(argv[++i]);
    }

    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        std::cerr << "Cannot read: " << source << std::endl;
        return 1;
    }

    YOLODetector detector(model, conf);
    auto dets = detector.detect(image);
    cv::Mat result = detector.draw(image, dets);

    std::string out_path = "output.jpg";
    cv::imwrite(out_path, result);
    std::cout << "Detections: " << dets.size() << ", saved: " << out_path << std::endl;
    return 0;
}
