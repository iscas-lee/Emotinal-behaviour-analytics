#include "stdafx.h"

using namespace ::std;
using namespace ::std::chrono;
using namespace ::OpenFace;

////////////////////////////////////////////////////////////////////////////////////////////////
///                                 PI / RAD-TO-DEG                                          ///
////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define RADTODEG(x)     ((x) * 180 / M_PI)

////////////////////////////////////////////////////////////////////////////////////////////////
///                                 WEBCAM SETTINGS                                          ///
////////////////////////////////////////////////////////////////////////////////////////////////
#define WEBCAM_WIDTH     640
#define WEBCAM_HEIGHT    480
#define WEBCAM_FPS       16.0

////////////////////////////////////////////////////////////////////////////////////////////////
///                                 OPENFACE++ FILES                                         ///
////////////////////////////////////////////////////////////////////////////////////////////////
#define BASEPATH_RESOURCES      "../../../../Engine/res/"

////////////////////////////////////////////////////////////////////////////////////////////////
///                                    OPENCV COLORS                                         ///
////////////////////////////////////////////////////////////////////////////////////////////////
const ::cv::Scalar COLOR_GREY  = ::cv::Scalar(100, 100, 100, false);
const ::cv::Scalar COLOR_WHITE = ::cv::Scalar(255, 255, 255, false);
const ::cv::Scalar COLOR_RED   = ::cv::Scalar(0,   0,   255, false);
const ::cv::Scalar COLOR_GREEN = ::cv::Scalar(0,   255, 0,   false);

////////////////////////////////////////////////////////////////////////////////////////////////
///                                    OTHER FIELDS                                          ///
////////////////////////////////////////////////////////////////////////////////////////////////
time_point<steady_clock> mTickLastMeasure;
time_point<steady_clock> mTick;
OscSender                mSender;

////////////////////////////////////////////////////////////////////////////////////////////////
///                                      SHIFTING                                            ///
////////////////////////////////////////////////////////////////////////////////////////////////
int shiftx;
int shifty;

enum Direction {
	ShiftUp = 1, ShiftRight, ShiftDown, ShiftLeft
};

cv::Mat shiftFrame(cv::Mat frame, int pixels, Direction direction)
{
	//create a same sized temporary Mat with all the pixels flagged as invalid (-1)
	cv::Mat temp = cv::Mat::zeros(frame.size(), frame.type());

	switch (direction)
	{
	case(ShiftUp):
		frame(cv::Rect(0, pixels, frame.cols, frame.rows - pixels)).copyTo(temp(cv::Rect(0, 0, temp.cols, temp.rows - pixels)));
		break;
	case(ShiftRight):
		frame(cv::Rect(0, 0, frame.cols - pixels, frame.rows)).copyTo(temp(cv::Rect(pixels, 0, frame.cols - pixels, frame.rows)));
		break;
	case(ShiftDown):
		frame(cv::Rect(0, 0, frame.cols, frame.rows - pixels)).copyTo(temp(cv::Rect(0, pixels, frame.cols, frame.rows - pixels)));
		break;
	case(ShiftLeft):
		frame(cv::Rect(pixels, 0, frame.cols - pixels, frame.rows)).copyTo(temp(cv::Rect(0, 0, frame.cols - pixels, frame.rows)));
		break;
	default:
		std::cout << "Shift direction is not set properly" << std::endl;
	}

	return temp;
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Appends a textline in libsvm format for given AU value set.
/// </summary>
void appendAU(ofstream& stream, ActionUnitValues& values, const char* svrval)
{
	stream 
		<< svrval << ' '
		<< "1:" << values.AU01 << ' ' 
		<< "2:" << values.AU02 << ' '
		<< "3:" << values.AU04 << ' '
		<< "4:" << values.AU05 << ' '
		<< "5:" << values.AU06 << ' '
		<< "6:" << values.AU07 << ' '
		<< "7:" << values.AU09 << ' '
		<< "8:" << values.AU10 << ' '
		<< "9:" << values.AU12 << ' '
		<< "10:" << values.AU14 << ' '
		<< "11:" << values.AU15 << ' '
		<< "12:" << values.AU17 << ' '
		<< "13:" << values.AU20 << ' '
		<< "14:" << values.AU23 << ' '
		<< "15:" << values.AU25 << ' '
		<< "16:" << values.AU26 << ' '
		<< "17:" << values.AU45 << ' '
		<< "\n";

	stream.flush();
}

/// <summary>
/// Handles a result dequeued from the Face System
/// </summary>
void processResult(::OpenFace::FaceSystem& system, ::OpenFace::Result& result, ::cv::Mat& statFrame)
{
	const int ROWDELTA = 18;

	///////////////////////////////////////////////////
	// send to neuromore Studio via OSC first
	mSender.send(result);

	///////////////////////////////////////////////////

	char formatted1[10];
	char formatted2[10];
	int x, y;

	::cv::rectangle(
		*result.image, cv::Point(0, 0), cv::Point(215, 30),
		cv::Scalar(0.0, 0.0, 0.0), CV_FILLED);

	::cv::putText(*result.image,
		"ANALYSER @ " + ::std::to_string((int)system.getFPS()) + " FPS",
		::cv::Point(5, 20),
		CV_FONT_HERSHEY_TRIPLEX,
		0.6, ::cv::Scalar(255, 255, 255, false));

	::cv::rectangle(
		*result.image, cv::Point(result.image->cols - 200, 0), cv::Point(result.image->cols, 30),
		cv::Scalar(0.0, 0.0, 0.0), CV_FILLED);

	sprintf_s(formatted1, "%01.f", 100.0 * result.certainty);
	::cv::putText(*result.image,
		"CERTAINTY @ " + ::std::string(formatted1) + " %",
		::cv::Point(result.image->cols - 195, 20),
		CV_FONT_HERSHEY_TRIPLEX,
		0.6, COLOR_WHITE);


	////// POSITION //////

	x = 5;
	y = 20;

	::cv::putText(statFrame,
		"POSITION",
		::cv::Point(x, y),
		CV_FONT_HERSHEY_TRIPLEX,
		0.6, COLOR_WHITE);

	y += 20;

	sprintf_s(formatted1, "%+06.1f", result.position[0]);
	::cv::putText(statFrame,
		"X: " + ::std::string(formatted1) + " mm",
		::cv::Point(x, y),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);
	
	y += 20;

	sprintf_s(formatted1, "%+06.1f", result.position[1]);
	::cv::putText(statFrame,
		"Y: " + ::std::string(formatted1) + " mm",
		::cv::Point(x, y),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	y += 20;

	sprintf_s(formatted1, "%+06.1f", result.position[2]);
	::cv::putText(statFrame,
		"Z: " + ::std::string(formatted1) + " mm",
		::cv::Point(x, y),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	////// ORIENTATION //////

	::cv::putText(statFrame,
		"ORIENTATION",
		::cv::Point(165, 20),
		CV_FONT_HERSHEY_TRIPLEX,
		0.6, COLOR_WHITE);

	sprintf_s(formatted1, "%+06.1f", RADTODEG(result.orientation[0]));
	::cv::putText(statFrame,
		"AX: " + ::std::string(formatted1) + " deg",
		::cv::Point(165, 40),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+06.1f", RADTODEG(result.orientation[1]));
	::cv::putText(statFrame,
		"AY: " + ::std::string(formatted1) + " deg",
		::cv::Point(165, 60),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+06.1f", RADTODEG(result.orientation[2]));
	::cv::putText(statFrame,
		"AZ: " + ::std::string(formatted1) + " deg",
		::cv::Point(165, 80),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	////// EYE GAZE LEFT //////

	::cv::putText(statFrame,
		"EYE GAZE",
		::cv::Point(325, 20),
		CV_FONT_HERSHEY_TRIPLEX,
		0.6, COLOR_WHITE);

	sprintf_s(formatted1, "%+06.3f", result.eyeLeft.gaze[0]);
	::cv::putText(statFrame,
		"LX: " + ::std::string(formatted1) + " ",
		::cv::Point(325, 40),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+06.3f", result.eyeLeft.gaze[1]);
	::cv::putText(statFrame,
		"LY: " + ::std::string(formatted1) + " ",
		::cv::Point(325, 60),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+06.3f", result.eyeLeft.gaze[2]);
	::cv::putText(statFrame,
		"LZ: " + ::std::string(formatted1) + " ",
		::cv::Point(325, 80),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+06.3f", result.eyeRight.gaze[0]);
	::cv::putText(statFrame,
		"RX: " + ::std::string(formatted1) + " ",
		::cv::Point(325, 100),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+06.3f", result.eyeRight.gaze[1]);
	::cv::putText(statFrame,
		"RY: " + ::std::string(formatted1) + " ",
		::cv::Point(325, 120),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+06.3f", result.eyeRight.gaze[2]);
	::cv::putText(statFrame,
		"RZ: " + ::std::string(formatted1) + " ",
		::cv::Point(325, 140),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	////// ATTENTION //////

	::cv::putText(statFrame,
		"ATTENTION",
		::cv::Point(485, 20),
		CV_FONT_HERSHEY_TRIPLEX,
		0.6, COLOR_WHITE);

	const double BARWIDTH = 100.0;

	::cv::rectangle(
		statFrame, cv::Point(485, 40), cv::Point(485 + (int)(BARWIDTH), 60),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(485, 40), cv::Point(485 + (int)(BARWIDTH * result.getAverageAttention()), 60),
		COLOR_WHITE, CV_FILLED);

	////// ACTION UNITS //////

	::cv::putText(statFrame,
		"ACTION UNITS",
		::cv::Point(640, 20),
		CV_FONT_HERSHEY_TRIPLEX,
		0.6, COLOR_WHITE);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU01);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU01);
	::cv::putText(statFrame,
		"AU01: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(640, 40),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU02);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU02);
	::cv::putText(statFrame,
		"AU02: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(640, 60),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU04);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU04);
	::cv::putText(statFrame,
		"AU04: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(640, 80),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU05);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU05);
	::cv::putText(statFrame,
		"AU05: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(640, 100),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU06);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU06);
	::cv::putText(statFrame,
		"AU06: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(640, 120),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU07);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU07);
	::cv::putText(statFrame,
		"AU07: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(640, 140),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU09);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU09);
	::cv::putText(statFrame,
		"AU09: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(800, 40),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU10);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU10);
	::cv::putText(statFrame,
		"AU10: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(800, 60),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU12);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU12);
	::cv::putText(statFrame,
		"AU12: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(800, 80),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU14);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU14);
	::cv::putText(statFrame,
		"AU14: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(800, 100),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU15);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU15);
	::cv::putText(statFrame,
		"AU15: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(800, 120),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU17);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU17);
	::cv::putText(statFrame,
		"AU17: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(800, 140),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU20);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU20);
	::cv::putText(statFrame,
		"AU20: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(960, 40),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU23);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU23);
	::cv::putText(statFrame,
		"AU23: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(960, 60),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU25);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU25);
	::cv::putText(statFrame,
		"AU25: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(960, 80),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU26);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU26);
	::cv::putText(statFrame,
		"AU26: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(960, 100),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	sprintf_s(formatted1, "%+1.3f", result.auSVR.AU45);
	sprintf_s(formatted2, "%1.f", result.auSVM.AU45);
	::cv::putText(statFrame,
		"AU45: " + ::std::string(formatted2) + " | " + ::std::string(formatted1),
		::cv::Point(960, 120),
		CV_FONT_HERSHEY_SIMPLEX,
		0.5, COLOR_RED);

	////// EMOTION FROM AU SVM + EXPRESSION //////

	x = 1120;
	y = 20;

	::cv::putText(statFrame, "EMOTIONS", ::cv::Point(x, y), CV_FONT_HERSHEY_TRIPLEX,
		0.6, COLOR_WHITE);

	y += ROWDELTA;

	::cv::putText(statFrame, "HAPPY", ::cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX,
		0.5, result.emotions.happy ? COLOR_GREEN : COLOR_GREY);

	y += ROWDELTA;

	::cv::putText(statFrame, "SAD", ::cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX,
		0.5, result.emotions.sad ? COLOR_GREEN : COLOR_GREY);

	y += ROWDELTA;

	::cv::putText(statFrame, "SURPRISED", ::cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX,
		0.5, result.emotions.surprised ? COLOR_GREEN : COLOR_GREY);

	y += ROWDELTA;

	::cv::putText(statFrame, "DISGUSTED", ::cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX,
		0.5, result.emotions.disgusted ? COLOR_GREEN : COLOR_GREY);

	y += ROWDELTA;

	::cv::putText(statFrame, "ANGRY", ::cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX,
		0.5, result.emotions.angry ? COLOR_GREEN : COLOR_GREY);

	y += ROWDELTA;

	::cv::putText(statFrame, "FEAR", ::cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX,
		0.5, result.emotions.feared ? COLOR_GREEN : COLOR_GREY);

	y += ROWDELTA;
	
	::cv::putText(statFrame, "NEUTRAL", ::cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX,
		0.5, result.emotions.neutral ? COLOR_GREEN : COLOR_GREY);
	
	////// EMOTION FROM LIBSVM SVR //////
	
	x = 1210;
	y = 20;
	const double EMOTBARWIDTH = 60.0;

	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH), y + 20),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH * result.emotionProbability.happy), y + 20),
		COLOR_WHITE, CV_FILLED);

	y += ROWDELTA;

	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH), y + 20),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH * result.emotionProbability.sad), y + 20),
		COLOR_WHITE, CV_FILLED);

	y += ROWDELTA;

	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH), y + 20),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH * result.emotionProbability.surprised), y + 20),
		COLOR_WHITE, CV_FILLED);

	y += ROWDELTA;

	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH), y + 20),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH * result.emotionProbability.disgusted), y + 20),
		COLOR_WHITE, CV_FILLED);

	y += ROWDELTA;

	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH), y + 20),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH * result.emotionProbability.angry), y + 20),
		COLOR_WHITE, CV_FILLED);

	y += ROWDELTA;

	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH), y + 20),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH * result.emotionProbability.feared), y + 20),
		COLOR_WHITE, CV_FILLED);

	y += ROWDELTA;

	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH), y + 20),
		cv::Scalar(100.0, 100.0, 100.0), CV_FILLED);
	::cv::rectangle(
		statFrame, cv::Point(x, y + 5), cv::Point(x + (int)(EMOTBARWIDTH * result.emotionProbability.neutral), y + 20),
		COLOR_WHITE, CV_FILLED);
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Main entry point
/// </summary>
int main(int argc, char** argv)
{
	bool optim = cv::useOptimized();

	bool ophv = cv::ocl::haveOpenCL();
	bool opcl = cv::ocl::useOpenCL();

	cv::setUseOptimized(true);
	
	
	////////////////////////////////////////////////////////////////////////////////////////////
	// create window
	::cv::namedWindow("FaceSensor", 1);
	::cv::setWindowTitle("FaceSensor", "FaceSensor");

	////////////////////////////////////////////////////////////////////////////////////////////
	// initialize vars
	::cv::Mat frameRaw;
	::cv::Mat outFrame;
	::cv::Mat statFrame;
	unsigned int countFrames = 0;
	unsigned int countFramesSkipped = 0;
	double fps;
	size_t pngcounter = 0;
	size_t argindex = 1;

	////////////////////////////////////////////////////////////////////////////////////////////
	// prepare output streams
	ofstream outputStream("au.txt", std::ofstream::out);
	ofstream outputStreamNeg("au_neg.txt", std::ofstream::out);

	////////////////////////////////////////////////////////////////////////////////////////////
	// open webcam or load image
	::cv::VideoCapture cap(CV_CAP_ANY);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT);
	cap.set(CV_CAP_PROP_FPS, WEBCAM_FPS);

	if (!cap.isOpened())
		return -1;

	////////////////////////////////////////////////////////////////////////////////////////////
	// create openface++ system
	FaceSystem mSystem(BASEPATH_RESOURCES);

	// start it
	mSystem.setIsRunning(true);

	////////////////////////////////////////////////////////////////////////////////////////////
	// Main Thread Loop
	for (;;)
	{
		// update current tick and delta to last tick
		mTick = high_resolution_clock::now();

		///////////////////////////////////////////////////////////////////////

		// get span since last tps measuring
		const duration<long long, nano> NANOS_SINCE_MEASURE =
			mTick - mTickLastMeasure;

		// possibly calculate new TPS value for last span (every ~1000ms)
		if (NANOS_SINCE_MEASURE >= milliseconds(1000))
		{
			const double RATIO = (double)countFrames / (double)NANOS_SINCE_MEASURE.count();
			fps = 1000000000.0 * RATIO;

			// reset counter and store last TPS update tick
			countFrames = 0;
			mTickLastMeasure = mTick;
		}

		///////////////////////////////////////////////////////////////////////

		if (argc >= 2)
		{
			frameRaw = ::cv::imread(argv[argindex], CV_LOAD_IMAGE_COLOR);
			frameRaw = shiftFrame(frameRaw, abs(shiftx), shiftx < 0 ? ShiftLeft : ShiftRight);
			frameRaw = shiftFrame(frameRaw, abs(shifty), shifty < 0 ? ShiftUp : ShiftDown);
			this_thread::sleep_for(milliseconds(16));
			
			// auto multi images
			if (argc > 2)
			{
				if (pngcounter >= 100)
				{
					pngcounter = 0;
					mSystem.reset();

					if (argc > argindex + 1)
						argindex++;

					else
						break;
				}
				else
					pngcounter++;
			}
		}
		else
		{
			// capture new frame
			// warning: this blocks until a new frame is avaible (limits to ~30 fps)
			cap >> frameRaw;
		}
		
		countFrames++;

		// allocate a new mat to store the image to enqueue
		::cv::Mat* frameCopy = new ::cv::Mat();
		frameRaw.copyTo(*frameCopy);

		// and try forward it to analyser
		if (!mSystem.enqueueWork(frameCopy))
		{
			// queue full, analyser busy
			countFramesSkipped++;
			delete frameCopy;
		}

		///////////////////////////////////////////////////////////////////////

		if (statFrame.cols != frameRaw.cols * 2)
			statFrame.create(160, frameRaw.cols * 2, frameRaw.type());
		
		statFrame.setTo(0);

		///////////////////////////////////////////////////////////////////////
		// get next processed frame from analyser
		::OpenFace::Result* result;
		if (result = mSystem.dequeueResult())
		{
			processResult(mSystem, *result, statFrame);

			// replace last analyser outputframe with new one
			result->image->copyTo(outFrame);

			// write AU to file
			if (result && argc > 2 && pngcounter == 95)
			{
				appendAU(outputStream, result->auSVR, "1.0");
				appendAU(outputStreamNeg, result->auSVR, "0.0");
			}
		}
		else
			result = nullptr;
		
		///////////////////////////////////////////////////////////////////////
		// Draw FPS and Skipped counter on camera image
		
		::cv::rectangle(
			frameRaw, cv::Point(0, 0), cv::Point(200, 30), 
			cv::Scalar(0.0, 0.0, 0.0), CV_FILLED);

		::cv::putText(frameRaw,
			"CAMERA @ " + ::std::to_string((int)fps) + " FPS",
			::cv::Point(5, 20),
			CV_FONT_HERSHEY_TRIPLEX,
			0.6, COLOR_WHITE);

		::cv::putText(frameRaw,
			"Skipped: " + ::std::to_string(countFramesSkipped),
			::cv::Point(20, 100),
			CV_FONT_HERSHEY_SIMPLEX,
			0.5, COLOR_RED);

		///////////////////////////////////////////////////////////////////////
		// combine and show images

		// add analyser output to the right
		if (outFrame.rows != 0 && outFrame.cols != 0)
			::cv::hconcat(frameRaw, outFrame, frameRaw);

		// add statistics to the bottom
		if (frameRaw.cols == statFrame.cols)
			::cv::vconcat(frameRaw, statFrame, frameRaw);

		// show it
		::cv::imshow("FaceSensor", frameRaw);

		///////////////////////////////////////////////////////////////////////
		// read keyboard input and process windows message queue

		int key = ::cv::waitKey(1) & 0xFF;
		switch (key)
		{
		case 97:  shiftx--; break;
		case 100: shiftx++; break;
		case 115: shifty--; break;
		case 119: shifty++; break;

		case 49: 
			if (result)
			{
				appendAU(outputStream, result->auSVR, "1.0");
				appendAU(outputStreamNeg, result->auSVR, "0.0");
			}
			break;

		case 27: return 0;	// ESC
		}

		//printf("%i\n", key);


		///////////////////////////////////////////////////////////////////////
		// cleanup analyser response heap allocs if there was one
		if (result)
		{
			delete result->image;
			delete result;
		}
	}

	outputStream.close();
	outputStreamNeg.close();

    return 0;
}
