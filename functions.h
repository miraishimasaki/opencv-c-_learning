//to implement various functions using opencv
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include<opencv2/tracking/tracking.hpp>
#include<opencv2/tracking/tracking_legacy.hpp>
#include<string>


using namespace cv;
using namespace std;

inline int camera_fail() {
	cout << "Cannot open camera" << endl;
	return -1;
}

inline int file_fail() {
	cout << "failed to open the image" << endl;
	return -1;
}

inline int frame_empty() {
	cout << "empty frame" << endl;
	return -1;
}


int facedetect() {
	CascadeClassifier face_cascade;
	// Load the cascade classifier,change the path to your own
	if (!face_cascade.load("D:\\source\\OpenCV4110\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")) {
		cout << "Error loading cascade classifier" << endl;
		return -1;
	}
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) camera_fail();
	Mat frame;
	while (true) {
		cap >> frame;
		if (frame.empty()) frame_empty();
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 3.0, 0, Size(30, 30));
		for (const auto& face : faces) {
			rectangle(frame, face, Scalar(123.255, 0), 5);
			putText(frame, "Face", Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0.255, 0), 2);
			cout << "Face detected:"<<(face.x+(face.width))/2<<"," <<((face.y)+(face.height))/2<< endl;			//to get the center of the face
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


//function to detect russian plates in an image
int platesdetect(string path,bool output= 0) {
	CascadeClassifier plate_cascade;
	// Load the cascade classifier,change the path to your own
	if (!plate_cascade.load("D:\\source\\OpenCV4110\\opencv\\sources\\data\\haarcascades\\haarcascade_russian_plate_number.xml")) {
		cout << "Error loading cascade classifier" << endl;
		return -1;
	}
	Mat img = imread(path, IMREAD_COLOR);
	resize(img, img, Size(640, 640));
	if (img.empty()) file_fail();
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	vector<Rect> plates;
	plate_cascade.detectMultiScale(gray, plates, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	for (Rect plate : plates) {
		rectangle(img, plate, Scalar(123.77, 0), 5);
	}
	imshow("Plates Detection", img);
	waitKey(0);
	if (output) {
		imwrite("car_result.jpg", img);
		cout << "result saved as car_result.jpg!" << endl;
	}
	return 0;
}

//function to detect edges in a live video using canny edge detection
int live_canny() {
	VideoCapture cap(0);
	if (!cap.isOpened()) camera_fail();
	Mat frame,edges;

	while (true) {
		cap >> frame;
		if (frame.empty()) frame_empty();
		
		Canny(frame, edges, 100, 200);
		imshow("Canny Edge Detection", edges);
		imshow("Original", frame);
		if (waitKey(1) == 27) {
			break;
			return 0;	
		}
	}
	cap.release();
	destroyAllWindows();
	return 0;
}


//function to overlay an image on another image using alpha blending
int ar_overlay(string sourse, string overlay,bool output= 0 ) {
	Mat img_src = imread(sourse, IMREAD_COLOR);
	Mat img_overlay = imread(overlay, IMREAD_UNCHANGED);

	if (img_src.empty() || img_overlay.empty()) file_fail();

	resize(img_overlay, img_overlay, Size(100, 100));

	Point center(img_src.cols / 2, img_src.rows / 2);

	// make sure the overlay image is at the center of the source image
	Mat roi = img_src(Rect(center.x - img_overlay.cols / 2, center.y - img_overlay.rows / 2, img_overlay.cols, img_overlay.rows));

	img_overlay.copyTo(roi, img_overlay);

	imshow("AR Overlay", img_src);

	waitKey(0);
	if (output) {
		imwrite("ar_result.jpg", img_src);
		cout << "result saved as ar_result.jpg!" << endl;
	}
	return 0;
	
}

//function to apply various filters to an image.1 Guassian blur 2 Canny edge 3 Gray filter
int filter(string path, int filter_type, bool output = 0) {
	Mat img = imread(path, IMREAD_COLOR);
	if (img.empty()) file_fail();
	Mat dst;
	switch (filter_type) {
	case 3:
		cvtColor(img, dst, COLOR_BGR2GRAY);
		break;
	case 2:
		Canny(img, dst, 100, 300);
		break;
	case 1:
		GaussianBlur(img, dst, Size(5, 5), 0);
		break;
	default:
		cout << "Invalid filter type!" << endl;
		return -1;
	}
	imshow("Filtered Image", dst);
	waitKey(0);
	if (output) {
		imwrite("filtered_result.jpg", dst);
		cout << "result saved as filtered_result.jpg!" << endl;
	}
	return 0;
}


//function to detect motion in a live video using absdiff and thresholding
int LiveMotionDetection() 
{
	VideoCapture cap(0);
	if (!cap.isOpened()) camera_fail();
	
	Mat frame, prevframe, diff;
	cap >> prevframe;
	if (prevframe.empty()) frame_empty();
	cvtColor(prevframe, prevframe, COLOR_BGR2GRAY);
	while(1){
		cap >> frame;
		if (frame.empty()) frame_empty();
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		absdiff(frame, prevframe, diff);
		threshold(diff, diff, 30, 255, THRESH_BINARY);//thresholding the difference image
		imshow("Motion Detection", diff);
		prevframe = frame.clone();
		if (waitKey(1) == 27) {
			break;
		}
		

	}
	cap.release();
	destroyAllWindows();
	return 0;
}

int ObjsTracking(string path="") {
	cv::VideoCapture cap;
	if (path == "") {
		cap.open(0);
	}
	else {
		cap.open(path);
	}
	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	cv::Mat frame;

	cap >> frame;
	//cv::resize(frame, frame, cv::Size(480, 480));
	if (frame.empty()) {
		cout << "Cannot read a frame from the video file" << endl;
		return -1;
	}
	cv::Ptr<cv::legacy::MultiTracker> multiTracker = cv::legacy::MultiTracker::create();
	vector<cv::Rect> bboxes;
	selectROIs("Select ROIs", frame, bboxes);
	for (const auto& bbox : bboxes) {
		multiTracker->add(cv::legacy::TrackerKCF::create(), frame, bbox);//KCF is the fastest  among all the trackers
	}
	while (true) {
		cap >> frame;

		if (frame.empty()) {
			break;
		}
		//cv::resize(frame, frame, cv::Size(480, 480));
		multiTracker->update(frame);
		for (const auto& bbox : multiTracker->getObjects()) {
			rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
		}
		cv::imshow("Multi-Object Tracking", frame);
		if (cv::waitKey(1) == 27) break; // ESC key to exit

	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

int video_division(string path, int num_parts) {
	VideoCapture cap(path);
	if (!cap.isOpened()) file_fail();
	Mat frame;
	int frame_count = 0;
	int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
	double fps;
	int width, height;
	fps = cap.get(cv::CAP_PROP_FPS);
	width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	//frame_count = get_frame(cap);
	frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
	int part_size = frame_count / num_parts;
	int start_frame = 0;
	int end_frame = part_size;
	for (int i = 0; i < num_parts; i++) {
		VideoWriter writer;
		string filename = "part" + to_string(i + 1) + ".avi";
		writer.open(filename, fourcc, fps, Size(width, height));
		if (!writer.isOpened()) {
			cout << "Cannot open the output video file" << endl;
			return -1;
		}
		cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
		while (true) {
			cap >> frame;
			if (frame.empty()) break;
			writer.write(frame);
			if (cap.get(cv::CAP_PROP_POS_FRAMES) == end_frame) { 
				cout << "Part " << i + 1 << " has been written to file" << endl;
				break;
			}
		}
		writer.release();
		start_frame = end_frame;
		end_frame += part_size;
		if (end_frame > frame_count) end_frame = frame_count;
	}
	cap.release();
	cout << "Video has been divided into " << num_parts << " parts" << endl;
	return 0;

}

//a poor method to get the frame count of a video
int get_frame(VideoCapture& cap) {
	Mat frame;
	int frame_count = 0;
	while (true) {
		cap >> frame;
		if (frame.empty()) { break; }
		frame_count++;
	}
	return frame_count;

}
	
	




	

