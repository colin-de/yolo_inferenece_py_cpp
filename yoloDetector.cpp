// 
// This is the minimal version of the yolo detector for showing detection results.
// 

#include <opencv2/opencv.hpp>
#include <fstream>
#include "ObjectDetector.h"

using namespace cv;
using namespace std;
using namespace cv::dnn;


int main() {
    // Create an object detector with the path to the model and the class list.
    ObjectDetector detector("../best_grey.onnx", "../fortiss.names");

    // Load an image.
    Mat frame = detector.loadImage("../img/1.png");

    // Store and print the detection result.
	detectionResult yolo_result;
	yolo_result = detector.detectResult(frame);

    return 0;
}

