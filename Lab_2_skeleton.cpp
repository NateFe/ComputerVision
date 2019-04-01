#include "stdafx.h"
#include <iostream>
#include <array>
#include <map>
#include <vector>

#include "OpenNI.h"
#include <NiTE.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace openni;
using namespace cv;
using namespace nite;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "../../haarcascade_frontalface_alt2.xml";
String eyes_cascade_name = "../../haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
//string window_name = "Capture - Face detection";
RNG rng(12345);
std::vector<Rect> faces; // faces variable initialization
//cv::Mat image = imread("file_name");  // uncomment when reading image. "file_name" also requires file extension (e.g. .jpg, .jpeg, .png). file should go in x64/Debug/
cv::Mat frame;
int cnt = 0; // use this variable to help count frames for face swap


int main(int argc, char** argv)
{
	bool flag = false;
	OpenNI::initialize();
	
	Device devAnyDevice;
	devAnyDevice.open(ANY_DEVICE);

	VideoStream streamDepth;
	streamDepth.create(devAnyDevice, SENSOR_DEPTH);

	VideoStream streamColor;
	streamColor.create(devAnyDevice, SENSOR_COLOR);

	VideoMode mModeDepth;
	mModeDepth.setResolution(640, 480);
	mModeDepth.setFps(30);
	mModeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
	streamDepth.setVideoMode(mModeDepth);


	VideoMode mModeColor;
	mModeColor.setResolution(640, 480);
	mModeColor.setFps(30);
	mModeColor.setPixelFormat(PIXEL_FORMAT_RGB888);
	streamColor.setVideoMode(mModeColor);

	if (devAnyDevice.isImageRegistrationModeSupported(
		IMAGE_REGISTRATION_DEPTH_TO_COLOR))
	{
		devAnyDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	}

	//check status, not important
	if (NiTE::initialize() != nite::STATUS_OK)
	{
		cerr << "NiTE initial error" << endl;
		return -1;
	}

	HandTracker mHandTracker;
	//check status, not important
	if (mHandTracker.create() != nite::STATUS_OK)
	{
		cerr << "Can't create user tracker" << endl;
		return -1;
	}
	
	mHandTracker.startGestureDetection(GESTURE_WAVE);
	mHandTracker.startGestureDetection(GESTURE_CLICK);
	//mHandTracker.startGestureDetection( GESTURE_HAND_RAISE );

	mHandTracker.setSmoothingFactor(0.1f);

	map< HandId, vector<cv::Point2f> > mapHandData;
	vector<cv::Point2f> vWaveList;
	vector<cv::Point2f> vClickList;
	cv::Point2f ptSize(3, 3);

	array<cv::Scalar, 8>    aHandColor;
	aHandColor[0] = cv::Scalar(255, 0, 0);
	aHandColor[1] = cv::Scalar(0, 255, 0);
	aHandColor[2] = cv::Scalar(0, 0, 255);
	aHandColor[3] = cv::Scalar(255, 255, 0);
	aHandColor[4] = cv::Scalar(255, 0, 255);
	aHandColor[5] = cv::Scalar(0, 255, 255);
	aHandColor[6] = cv::Scalar(255, 255, 255);
	aHandColor[7] = cv::Scalar(0, 0, 0);



	streamDepth.start();
	streamColor.start();


	//namedWindow("Depth Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Color Image", CV_WINDOW_AUTOSIZE);


	int iMaxDepth = streamDepth.getMaxPixelValue();


	VideoFrameRef  frameDepth;
	VideoFrameRef  frameColor;


	while (true)
	{
		// create Mat for color image
		cv::Mat cImageBGR;

		// read color stream
		VideoFrameRef mColorFrame;
		streamColor.readFrame(&mColorFrame);

		// convert raw RGB data to OpenCV data, in CV_8UC3
		const cv::Mat mImageRGB(mColorFrame.getHeight(), mColorFrame.getWidth(),
			CV_8UC3, (void*)mColorFrame.getData());

		// RGB ==> BGR
		cv::cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);


		// get hand frame
		HandTrackerFrameRef mHandFrame;
		if (mHandTracker.readFrame(&mHandFrame) == nite::STATUS_OK)
		{
			openni::VideoFrameRef mDepthFrame = mHandFrame.getDepthFrame();
			// convert depth raw to opencv
			const cv::Mat mImageDepth(mDepthFrame.getHeight(), mDepthFrame.getWidth(), CV_16UC1, (void*)mDepthFrame.getData());
			// to make depth clearer
			cv::Mat mScaledDepth, mImageBGR;
			mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / 10000);
			// cover grayscale to RGB, for drawing point and track
			cv::cvtColor(mScaledDepth, mImageBGR, CV_GRAY2BGR);

			// detect hand
			const nite::Array<GestureData>& aGestures = mHandFrame.getGestures();
			for (int i = 0; i < aGestures.getSize(); ++i)
			{
				const GestureData& rGesture = aGestures[i];
				const nite::Point3f& rPos = rGesture.getCurrentPosition();
				cv::Point2f rPos2D;
				mHandTracker.convertHandCoordinatesToDepth(rPos.x, rPos.y, rPos.z, &rPos2D.x, &rPos2D.y);

				// draw point
				switch (rGesture.getType())
				{
				case GESTURE_WAVE:
					vWaveList.push_back(rPos2D);
					//print gesture recognized
					cout << "Waving gesture recognized" << endl;
					//switch flag to 'on', which turns on classifier
					flag = !flag;
					break;

				case GESTURE_CLICK:
					vClickList.push_back(rPos2D);
					//print gesture recognized
					cout << "Clicking gesture recognized" << endl;
					//switch flag to 'on', which turns on classifier
					flag = !flag;
					break;
				}

				// track hand
				HandId mHandID;
				if (mHandTracker.startHandTracking(rPos, &mHandID) != nite::STATUS_OK)
					cerr << "Can't track hand" << endl;
			}

			// get hand coordinate
			const nite::Array<HandData>& aHands = mHandFrame.getHands();
			for (int i = 0; i < aHands.getSize(); ++i)
			{
				const HandData& rHand = aHands[i];
				HandId uID = rHand.getId();

				if (rHand.isNew())
				{
					mapHandData.insert(make_pair(uID, vector<cv::Point2f>()));
				}

				if (rHand.isTracking())
				{
					// put hand coordinate to depth and RGB
					const nite::Point3f& rPos = rHand.getPosition();
					cv::Point2f rPos2D;
					mHandTracker.convertHandCoordinatesToDepth(rPos.x, rPos.y, rPos.z, &rPos2D.x, &rPos2D.y);

					mapHandData[uID].push_back(rPos2D);
				}

				if (rHand.isLost())
					mapHandData.erase(uID);
			}

			// draw point and track
			for (auto itHand = mapHandData.begin(); itHand != mapHandData.end(); ++itHand)
			{
				const cv::Scalar& rColor = aHandColor[itHand->first % aHandColor.size()];
				const vector<cv::Point2f>& rPoints = itHand->second;

				for (int i = 1; i < rPoints.size(); ++i)
				{
					cv::line(mImageBGR, rPoints[i - 1], rPoints[i], rColor, 2);
					cv::line(cImageBGR, rPoints[i - 1], rPoints[i], rColor, 2);
				}
			}

			// draw click gesture
			for (auto itPt = vClickList.begin(); itPt != vClickList.end(); ++itPt)
			{
				cv::circle(mImageBGR, *itPt, 5, cv::Scalar(0, 0, 255), 2);
				cv::circle(cImageBGR, *itPt, 5, cv::Scalar(0, 0, 255), 2);
			}

			// draw wave gesture
			for (auto itPt = vWaveList.begin(); itPt != vWaveList.end(); ++itPt)
			{
				cv::rectangle(mImageBGR, *itPt - ptSize, *itPt + ptSize, cv::Scalar(0, 255, 0), 2);
				cv::rectangle(cImageBGR, *itPt - ptSize, *itPt + ptSize, cv::Scalar(0, 255, 0), 2);
			}

			//show image
			//cv::imshow("Depth Image", mImageBGR);
			cv::imshow("Color Image", cImageBGR); cv::moveWindow("Color Image", 1920 - (640 + 20), 0);

			mHandFrame.release();
		}
		else
		{
			cerr << "Can't get new frame" << endl;
		}
		
		
		//-- 1. Load the cascades
		if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
		//if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };



		//-- 2. Apply the classifier to the frame
		if (!cImageBGR.empty() && flag)
		{
			detectAndDisplay(cImageBGR);
			faces.clear();
		}



		char key = waitKey(1);

		if (key == 's') {
			cout << "Switch flag key" << endl;
			flag = !flag;
		}

	}


	streamDepth.destroy();
	streamColor.destroy();


	devAnyDevice.close();


	openni::OpenNI::shutdown();

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame_input)
{

	Mat frame_gray;
	frame_input.copyTo(frame);
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces by calling detectMultiScale
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));


	for (size_t i = 0; i < faces.size(); i++) // loop through all detected faces
	{
		//drawing circle around face
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		////Gaussian blur code here -- comment out when moving to face swap portion

		


		////face swap code here -- comment out if you want to redo gaussian blur
		


	}
	//-- Show what you got
	imshow("Color Image", frame);
}