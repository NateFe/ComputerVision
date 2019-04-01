#include "stdafx.h"
#include <iostream>
#include "OpenNI.h"

// porting OpenCV library 
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace openni;
using namespace cv;

cv::Mat gDataDepth;
//you can save data in these variables
int x_one = 0;
int y_one = 0;
int x_two = 0;
int y_two = 0;
double depth_one = 0;
double depth_two = 0;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{

	//The goal of the lab is to find the coefficients a and b that will convert the variable result
	//into depth (in centermeters); find the focal length of the length; and use triangluar similarity
	//formula (with the calculated width in pixel from program, depth, and focal length) to find 
	//the object size from camera view, in centermeters.

	//this skeleton provides the basic framework for looking at the result variable by clicking mouse buttons,
	//you can gather two points from left click and right click, and generate a line in pixel, where you a
	Scalar intensity = gDataDepth.at<uchar>(y, x);
	double result = intensity.val[0];
	//double depth = 4*result + 3; // you'll probably get a relationship similar to this one, but change your coefficients to match your fit

	double focal = 0; //calculate focal length and assign it here

	if (event == EVENT_LBUTTONDOWN)
	{
		x_one = x;
		y_one = y;
		//depth_one = depth;
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << "), depth (data value " << result << ")" << endl;
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		x_two = x;
		y_two = y;
		//depth_two = depth;
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << "), depth (data value " << result << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		// width measurements go in here 
	}
	else if (event == EVENT_MOUSEMOVE) // for debugging purposes, uncomment the line of code in this else if statement
	{
		//cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

	}

}

int main(int argc, char** argv)
{
	// Initialize OpenNI environment 
	OpenNI::initialize();

	// construct and open Device，which is Kinect (openNI also works for other devices), ANY_DEVICE flag works for a single device setup
	Device devAnyDevice;
	devAnyDevice.open(ANY_DEVICE);

	// create depth stream
	VideoStream streamDepth;
	streamDepth.create(devAnyDevice, SENSOR_DEPTH);

	// create RGB stream
	VideoStream streamColor;
	streamColor.create(devAnyDevice, SENSOR_COLOR);

	//set the viewing mode for depth stream
	VideoMode mModeDepth;
	// set resolution
	mModeDepth.setResolution(640, 480);
	// set FPS
	mModeDepth.setFps(30);
	// Pixel Format
	mModeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
	//call the funciton in the streamDepth construct to set Video Mode
	streamDepth.setVideoMode(mModeDepth);

	// Same as depth stream
	VideoMode mModeColor;
	mModeColor.setResolution(640, 480);
	mModeColor.setFps(30);
	mModeColor.setPixelFormat(PIXEL_FORMAT_RGB888);

	streamColor.setVideoMode(mModeColor);

	devAnyDevice.setDepthColorSyncEnabled(true);

	// Image Registration Mode, first check if image resigstration is supported in this device in the if bracket
	devAnyDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	// start both streams
	streamDepth.start();
	streamColor.start();



	// create OpenCV windows
	namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
	moveWindow("Depth Image", 1920 - (640 + 20), 0);
	namedWindow("Color Image", CV_WINDOW_AUTOSIZE);
	moveWindow("Color Image", 1920 - (640 + 20), 500);

	setMouseCallback("Depth Image", CallBackFunc, NULL);
	//setMouseCallback("Color Image", CallBackFunc, NULL);


	// get maximum pixel value from depth stream
	int iMaxDepth = streamDepth.getMaxPixelValue();

	// Create a loop, streams read frame from variable construct VideoFrameRef
	VideoFrameRef  frameDepth;
	VideoFrameRef  frameColor;

	while (true)
	{
		// read data stream
		streamDepth.readFrame(&frameDepth);
		streamColor.readFrame(&frameColor);

		// change raw depth data to OpenCV datatype - cv::Mat
		const cv::Mat mImageDepth(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1, (void*)frameDepth.getData());
		// to make the depth stream look better, convert CV_16UC1 ==> CV_8U format
		cv::Mat mScaledDepth;
		mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / iMaxDepth);

		// imshow shows the depth stream data in "Depth Image" window
		cv::imshow("Depth Image", mScaledDepth);
		gDataDepth = mScaledDepth;

		// change raw image data to OpenCV datatype - cv::Mat
		const cv::Mat mImageRGB(frameColor.getHeight(), frameColor.getWidth(), CV_8UC3, (void*)frameColor.getData());
		// openCV shows image in BGR not RGB, so we are converting raw RGB data to BGR
		cv::Mat cImageBGR;
		cv::cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);
		// imshow shows the image stream data
		cv::imshow("Color Image", cImageBGR);

		// catch key to break the loop
		if (cv::waitKey(1) == 'q')
			break;
	}

	// close data stream
	streamDepth.destroy();
	streamColor.destroy();

	// close device
	devAnyDevice.close();

	// close OpenNI interface
	openni::OpenNI::shutdown();

	return 0;
}



