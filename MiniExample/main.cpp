/***************************************************************************/
/*************************** OpenFace++ Example ****************************/
/***************************************************************************/

// includes
#include <FaceSystem.h>

// used namespaces
using namespace ::std;
using namespace ::OpenFace;

// window title etc.
#define APPNAME "OpenFace++ Example"

// basefolder for resources
#define RES "../../../../Engine/res/"
 
/////////////////////////////////////////////////////////////////////////////
// Main entry point
/////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{	
	// create window
	::cv::namedWindow(APPNAME, 1);
	::cv::setWindowTitle(APPNAME, APPNAME);

	// open webcam
	::cv::VideoCapture cap(CV_CAP_ANY);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// failed to open webcam
	if (!cap.isOpened())
		return -1;

	// create openface++ system
	FaceSystem mSystem(RES);

	// start openface++ system
	mSystem.setIsRunning(true);
	
	// enter thread loop
	for (;;)
	{		
		// allocate a new mat to enqueue later
		::cv::Mat* frame = new ::cv::Mat();

		// capture a new frame
		// warning: this blocks until a new frame is available 
		// typically limits loops to webcam fps @ ~15-60 fps.
		cap >> *frame;
			
		// try to forward the new frame to face system
		// if queue is full then analyser is busy, so skip frame
		if (!mSystem.enqueueWork(frame))		
			delete frame;
		
		// try to get last processed frame from face system
		// if available, show and delete it
		if (Result* result = mSystem.dequeueResult())
		{	
			::cv::imshow(APPNAME, *result->image);
			delete result->image;
			delete result;
		}

		// use opencv to process windows message queue
		int key = ::cv::waitKey(1) & 0xFF;		
	}

	// normal exit
    return 0;
}
