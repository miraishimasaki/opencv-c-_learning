//to implement various functions using opencv


#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int facedetect() {
	CascadeClassifier face_cascade;
	// Load the cascade classifier,change the path to your own
	if (!face_cascade.load("D:\\source\\OpenCV4110\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")) {
		cout << "Error loading cascade classifier" << endl;
		return -1;
	}
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) {
		cout << "Cannot open camera" << endl;
		return -1;
	}
	Mat frame;
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			break;
			return -1;
		}
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 3.0, 0, Size(30, 30));
		for (const auto& face : faces) {
			rectangle(frame, face, Scalar(0.255, 0), 5);
			putText(frame, "Face", Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0.255, 0), 2);
			//cout << "Face detected:"<<face.x<<"," <<face.y<< endl;
		}
		imshow("Face Detection", frame);
		if (waitKey(1) == 27) {
			break;
			return -1;
		}

	}
	cap.release();
	destroyAllWindows();
	return 0;
}

