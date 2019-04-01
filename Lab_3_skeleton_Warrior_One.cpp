#include "stdafx.h"
#include <iostream>

// OpenCV head files
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <OpenNI.h>
#include <NiTE.h>

using namespace std;
using namespace openni;
using namespace nite;


long double angle_degree(cv::Point2f pt_one, cv::Point2f pt_two);
//string file_name = "test.oni";

int main(int argc, char **argv)
{
	bool display_name = true;
	bool detect_shape = true;
	bool detect_shape_two = false;
	cv::Point2f display_position_1(1, 440);
	cv::Point2f display_position_2(1, 440);
	// initialize OpenNI
	OpenNI::initialize();

	// Open Kinect device
	Device  mDevice;
	mDevice.open(ANY_DEVICE);

	// create depth stream
	VideoStream mDepthStream;
	mDepthStream.create(mDevice, SENSOR_DEPTH);

	// set Video Mode
	VideoMode mDepthMode;
	mDepthMode.setResolution(640, 480);
	mDepthMode.setFps(30);
	mDepthMode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
	mDepthStream.setVideoMode(mDepthMode);

	// create color stream
	VideoStream mColorStream;
	mColorStream.create(mDevice, SENSOR_COLOR);
	// same setting for Color Stream
	VideoMode mColorMode;
	mColorMode.setResolution(640, 480);
	mColorMode.setFps(30);
	mColorMode.setPixelFormat(PIXEL_FORMAT_RGB888);
	mColorStream.setVideoMode(mColorMode);

	// impose the depth view onto the color view
	mDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	// To get skeleton data, initialize NiTE
	NiTE::initialize();

	// Create Usertracker variable
	UserTracker mUserTracker;
	mUserTracker.create(&mDevice);

	// Control the smoothing factor of the skeleton joints. Factor should be between 0 (no smoothing at all) and 1 (no movement at all), maybe change this dynamically?
	mUserTracker.setSkeletonSmoothingFactor(0.5f);

	// create window for user image
	cv::namedWindow("User Image", CV_WINDOW_AUTOSIZE);

	int iMaxDepth = mDepthStream.getMaxPixelValue();
	// start running streams
	mDepthStream.start();
	mColorStream.start();

	while (true)
	{
		// create OpenCV：：Mat for color stream
		cv::Mat cImageBGR;

		// read color stream in frames
		VideoFrameRef mColorFrame;
		mColorStream.readFrame(&mColorFrame);

		// convert frame data to opencv format
		const cv::Mat mImageRGB(mColorFrame.getHeight(), mColorFrame.getWidth(),
			CV_8UC3, (void*)mColorFrame.getData());

		// RGB ==> BGR
		cv::cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);

		// read usertracker stream in frames
		UserTrackerFrameRef  mUserFrame;
		mUserTracker.readFrame(&mUserFrame);

		// gather user information
		const nite::Array<UserData>& aUsers = mUserFrame.getUsers();
		for (int i = 0; i < aUsers.getSize(); ++i)
		{
			const UserData& rUser = aUsers[i];

			// check user status
			if (rUser.isNew())
			{
				// start skeleton tracking
				mUserTracker.startSkeletonTracking(rUser.getId());
			}

			if (rUser.isVisible())
			{
				// if visible, get skeleton data
				const Skeleton& rSkeleton = rUser.getSkeleton();

				// check if skeleton data status is "tracking"
				if (rSkeleton.getState() == SKELETON_TRACKED)
				{
					// get the 15 joints 
					SkeletonJoint aJoints[15];
					aJoints[0] = rSkeleton.getJoint(JOINT_HEAD);
					aJoints[1] = rSkeleton.getJoint(JOINT_NECK);
					aJoints[2] = rSkeleton.getJoint(JOINT_LEFT_SHOULDER);
					aJoints[3] = rSkeleton.getJoint(JOINT_RIGHT_SHOULDER);
					aJoints[4] = rSkeleton.getJoint(JOINT_LEFT_ELBOW);
					aJoints[5] = rSkeleton.getJoint(JOINT_RIGHT_ELBOW);
					aJoints[6] = rSkeleton.getJoint(JOINT_LEFT_HAND);
					aJoints[7] = rSkeleton.getJoint(JOINT_RIGHT_HAND);
					aJoints[8] = rSkeleton.getJoint(JOINT_TORSO);
					aJoints[9] = rSkeleton.getJoint(JOINT_LEFT_HIP);
					aJoints[10] = rSkeleton.getJoint(JOINT_RIGHT_HIP);
					aJoints[11] = rSkeleton.getJoint(JOINT_LEFT_KNEE);
					aJoints[12] = rSkeleton.getJoint(JOINT_RIGHT_KNEE);
					aJoints[13] = rSkeleton.getJoint(JOINT_LEFT_FOOT);
					aJoints[14] = rSkeleton.getJoint(JOINT_RIGHT_FOOT);

					// convert the 3D data coordinates to 2D for plotting purpose, this is relative to depth window but we turned on image registration so this is good
					cv::Point2f aPoint[15];
					cv::Point3f bPoint[15];
					for (int s = 0; s < 15; ++s)
					{
						const Point3f& rPos = aJoints[s].getPosition();
						bPoint[s].x = rPos.x;
						bPoint[s].y = rPos.y;
						bPoint[s].z = (int) (255.0 * rPos.z / iMaxDepth);
						mUserTracker.convertJointCoordinatesToDepth(
							rPos.x, rPos.y, rPos.z,
							&(aPoint[s].x), &(aPoint[s].y));
					}

					// draw lines between the joints
					cv::line(cImageBGR, aPoint[0], aPoint[1], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[1], aPoint[2], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[1], aPoint[3], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[2], aPoint[4], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[3], aPoint[5], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[4], aPoint[6], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[5], aPoint[7], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[1], aPoint[8], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[8], aPoint[9], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[8], aPoint[10], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[9], aPoint[11], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[10], aPoint[12], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[11], aPoint[13], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[12], aPoint[14], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[9], aPoint[10], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[2], aPoint[8], cv::Scalar(255, 0, 0), 2);
					cv::line(cImageBGR, aPoint[3], aPoint[8], cv::Scalar(255, 0, 0), 2);

					if (display_name == true) {
						cv::putText(cImageBGR, "head", aPoint[0], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "neck", aPoint[1], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "left shoulder", aPoint[2], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "right shoulder", aPoint[3], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "left elbow", aPoint[4], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "right elbow", aPoint[5], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "left hand", aPoint[6], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "right hand", aPoint[7], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "torso", aPoint[8], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "left hip", aPoint[9], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "right hip", aPoint[10], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "left knee", aPoint[11], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "right knee", aPoint[12], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "left foot", aPoint[13], 1, 1, cv::Scalar(0, 255, 255), 1);
						cv::putText(cImageBGR, "right foot", aPoint[14], 1, 1, cv::Scalar(0, 255, 255), 1);
						
					}

					if (detect_shape == true) {
						//detect_shape is by default true, and can be inverted to false, by pressing 'd' on your keyboard,
						//while the window is selected

						//here below is an example of how you can implement this method

						//all the joints are listed in the variables aPoint[i], you can set your constraints on how bodies 
						//should be positioned by many ways, e.g. set a constraint on the x difference/y difference on two points,
						//or, you can use the given method called 'angle_degree(Point 1, Point 2)' which will return the angle between
						//two points, with respect to the horizontal axis - i.e. angle_degree((0,0),(sqrt(3),1)) returns 30 degrees

						//the method cv::putText is used to put text on an image, the behavior is defined as 
						//cv::putText(Mat video_frame, string text, Point display_position, font type, font size, color of text, thickness)

						//use 'occupied' as a flag to see if the the display_position is taken to display something, you can modify it
						//if you want to display multiple things at once - in which case you may want to define multiple display_positions
						//to be used
						//if 'occupied' is switched to one, there is text on the screen, o.w. not;
						//if 'occupied' is zero still at the end, display motion complete

						//obviously, you cannot use Warrior 2 as your pose, since part of it is shown here already
						int occupied = 0;

						if (abs(aPoint[4].y - aPoint[6].y) > 20) {

							if (abs(aPoint[2].y - aPoint[4].y) > 20) {
								occupied = 1;
								cv::putText(cImageBGR, "left shoulder and elbow not aligned", display_position_1, 1, 2, cv::Scalar(255, 0, 0), 2);


							}
							if (occupied == 0) {
								occupied = 1;
								cv::putText(cImageBGR, "left hand and elbow not aligned", display_position_1, 1, 2, cv::Scalar(255, 0, 0), 2);
							}
						}

						if (angle_degree(aPoint[9], aPoint[11]) > 45) {
							if (occupied == 0) {
								occupied = 1;
								cv::putText(cImageBGR, "left hip and left knee at relatively even level", display_position_1, 1, 2, cv::Scalar(0, 0, 255), 2);
							}

						}

						if (occupied == 0) {
							cv::putText(cImageBGR, "Warrior two complete", display_position_1, 1, 2, cv::Scalar(0, 255, 0), 2);

						}


						

					}

					if (detect_shape_two == true) {
						int occupied = 0;





					}
					// also draw circles around joints
					// if position confidence is great, the circle is drawn in green, otherwise red
					for (int s = 0; s < 15; ++s)
					{
						if (aJoints[s].getPositionConfidence() > 0.5)
							cv::circle(cImageBGR, aPoint[s], 3, cv::Scalar(0, 255, 0), 3);
						else
							cv::circle(cImageBGR, aPoint[s], 3, cv::Scalar(0, 0, 255), 3);
					}
				}
			}
		}

		// display image
		cv::imshow("User Image", cImageBGR);

		// press q to break loop
		char key = cv::waitKey(1);

		if (key == 's') {
			cout << "Switch display key" << endl;
			display_name = !display_name;
		}
		if (key == 'd') {
			cout << "Switch detect key" << endl;
			detect_shape = !detect_shape;
			detect_shape_two = !detect_shape_two;
		}
		if (key == 'q') {
			break;
		}
	}

	// first turn off usertracker
	mUserTracker.destroy();

	// turn off streams
	mColorStream.destroy();
	mDepthStream.destroy();

	// close kinect
	mDevice.close();

	// close Nite and OpenNI
	NiTE::shutdown();
	OpenNI::shutdown();

	return 0;
}

long double angle_degree(cv::Point2f pt_one, cv::Point2f pt_two) {

	double diff_x = abs(pt_one.x - pt_two.x);
	double diff_y = abs(pt_one.y - pt_two.y);

	long double angle_in_degree = (180 / 3.1415926)*atan2(diff_y, diff_x);

	return angle_in_degree;

}