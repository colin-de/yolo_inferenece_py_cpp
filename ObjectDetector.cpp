// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

class ObjectDetector {
    // Constants.
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 480.0;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.60;

    // Text parameters.
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 1;

    // Colors.
    Scalar BLACK = Scalar(0,0,0);
    Scalar BLUE = Scalar(255, 178, 50);
    Scalar YELLOW = Scalar(0, 255, 255);
    Scalar RED = Scalar(0,0,255);

    Net net;
    vector<string> class_list;

public:
    ObjectDetector(const string& model_path, const string& class_list_path) {
        ifstream ifs(class_list_path);
        string line;
        while (getline(ifs, line)) {
            class_list.push_back(line);
        }

        net = readNet(model_path);
    }

    Mat loadImage(const string& image_path) {
        return imread(image_path);
    }

    Mat detect(Mat& frame) {
        vector<Mat> detections = pre_process(frame);
        Mat cloned_frame = frame.clone();
        return post_process(cloned_frame, detections, class_list);
    }

private:
    void draw_label(Mat& input_image, const string& label, int left, int top) {
        int baseLine;

        Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);

        top = max(top, label_size.height);

        Point tlc = Point(left, top);

        Point brc = Point(left + label_size.width, top + label_size.height + baseLine);

        rectangle(input_image, tlc, brc, BLACK, FILLED);

        putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
    }

    vector<Mat> pre_process(Mat &input_image) {
        // Convert to blob.
        Mat blob;
        blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

        net.setInput(blob);

        // Forward propagate.
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        return outputs;
    }

    Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name) {
        // Initialize vectors to hold respective outputs while unwrapping detections.
        vector<int> class_ids;
        vector<float> confidences;
        vector<Rect> boxes;

        // Resizing factor.
        float x_factor = input_image.cols / INPUT_WIDTH;
        float y_factor = input_image.rows / INPUT_HEIGHT;

        float *data = (float *)outputs[0].data;

        //    const int dimensions = 85;
        const int dimensions = 8;
        const int rows = 25200;
        // Iterate through 25200 detections.
        for (int i = 0; i < rows; ++i)
        {
            float confidence = data[4];
            // Discard bad detections and continue.
            if (confidence >= CONFIDENCE_THRESHOLD)
            {
                float * classes_scores = data + 5;
                // Create a 1x85 Mat and store class scores of 80 classes.
                Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
                // Perform minMaxLoc and acquire index of best class score.
                Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                // Continue if the class score is above the threshold.
                if (max_class_score > SCORE_THRESHOLD)
                {
                    // Store class ID and confidence in the pre-defined respective vectors.

                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    // Center.
                    float cx = data[0];
                    float cy = data[1];
                    // Box dimension.
                    float w = data[2];
                    float h = data[3];
                    // Bounding box coordinates.
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    // Store good detections in the boxes vector.
                    boxes.emplace_back(left, top, width, height);
                }

            }
            // Jump to the next column.
            //        data += 85;
            data += 8;
        }

        // Perform Non Maximum Suppression and draw predictions.
        vector<int> indices;
        NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        std::ofstream outfile("output.txt");

        std::map<int, std::tuple<std::string, std::string, float, std::string>> best_detections;
        const float MAX_CONFIDENCE = 1.0;

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            Rect box = boxes[idx];

            int left = box.x;
            int top = box.y;
            int width = box.width;
            int height = box.height;
            float confidence = confidences[idx];

            // Draw bounding box.
            rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

            // Define box corners
            Point topLeft = Point(left, top);
            string topLeftStr = "[" + std::to_string(topLeft.x) + ", " + std::to_string(topLeft.y) + "]";

            Point bottomRight = Point(left + width, top + height);
            string bottomRightStr = "[" + std::to_string(bottomRight.x) + ", " + std::to_string(bottomRight.y) + "]";

            // Get the label for the class name and its confidence.
            string label = format("%.2f", confidences[idx]);
            // std::cout << topLeftStr << " " << bottomRightStr << " " << class_ids[idx] << " " << class_name[class_ids[idx]] << std::endl;
            int class_id = class_ids[idx];
            string class_name_str = class_name[class_ids[idx]];
            // label = class_name[class_ids[idx]] + ":" + label; // Uncomment this line to display class name and confidence.

            // Draw class labels.
            // draw_label(input_image, label, left, top); // Uncomment this line to display class name and confidence.

            if (best_detections.find(class_id) == best_detections.end()) {
                if (confidence <= MAX_CONFIDENCE && confidence > CONFIDENCE_THRESHOLD)
                    best_detections[class_id] = std::make_tuple(topLeftStr, bottomRightStr, confidence, label);
            }
            else if (confidence > std::get<2>(best_detections[class_id]) && confidence <= MAX_CONFIDENCE) {
                best_detections[class_id] = std::make_tuple(topLeftStr, bottomRightStr, confidence, label);
            }
        }
        std::cout << "Best detections: " << std::endl;
        for (const auto& kvp : best_detections) {
        //        std::cout << "Label: " << std::get<3>(kvp.second) << std::endl;
            std::cout << std::get<0>(kvp.second) << " " << std::get<1>(kvp.second) << " " << kvp.first << " " << class_name[kvp.first] << std::endl;
            std::cout << "Confidence: " << std::get<2>(kvp.second) << std::endl;
        }
        outfile.close();
        return input_image;
    }
};


int main() {
    // Create an object detector with the path to the model and the class list.
    ObjectDetector detector("../best.onnx", "../fortiss.names");

    // Load an image.
    Mat frame = detector.loadImage("../000000.png");

    // Detect objects in the image.
    Mat result = detector.detect(frame);

    // Display the result.
    imshow("Output", result);
    waitKey(0);

    return 0;
}

