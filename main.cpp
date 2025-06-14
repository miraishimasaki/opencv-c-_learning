#include "functions.h"
int main() {
//	int status;
//	//status = facedetect();
//	status = LiveMotionDetection();
//	//status = live_canny();
//	//status = ar_overlay("car.jpg","berserk.jpg");
//	return status;
	int status;
	string video_path;
	cout << "Enter the path of the video to be divided: ";
	cin >> video_path;
	status = video_division(video_path,4);
}


