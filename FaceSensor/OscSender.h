#pragma once

#include <openface\FaceWorker.h>
#include <openface\FaceResult.h>
#include <oscpack\OscOutboundPacketStream.h>
#include <oscpack\UdpSocket.h>

#define OSC_DEFAULT_ADDRESS     "127.0.0.1"
#define OSC_DEFAULT_PORT        4545
#define OSC_OUTPUT_BUFFER_SIZE  4096

#define OSC_ADDR_FACE_DETECTED	        "/face/detected"
#define OSC_ADDR_FACE_CERTAINTY         "/face/certainty"
#define OSC_ADDR_FACE_POSITION_X	    "/face/position/x"
#define OSC_ADDR_FACE_POSITION_Y	    "/face/position/y"
#define OSC_ADDR_FACE_POSITION_Z	    "/face/position/z"
#define OSC_ADDR_FACE_ORIENTATION_AX    "/face/orientation/ax"
#define OSC_ADDR_FACE_ORIENTATION_AY    "/face/orientation/ay"
#define OSC_ADDR_FACE_ORIENTATION_AZ    "/face/orientation/az"
#define OSC_ADDR_FACE_EYEGAZE_LEFT_X    "/face/eyegaze/left/x"
#define OSC_ADDR_FACE_EYEGAZE_LEFT_Y    "/face/eyegaze/left/y"
#define OSC_ADDR_FACE_EYEGAZE_LEFT_Z    "/face/eyegaze/left/z"
#define OSC_ADDR_FACE_EYEGAZE_RIGHT_X   "/face/eyegaze/right/x"
#define OSC_ADDR_FACE_EYEGAZE_RIGHT_Y   "/face/eyegaze/right/y"
#define OSC_ADDR_FACE_EYEGAZE_RIGHT_Z   "/face/eyegaze/right/z"
#define OSC_ADDR_FACE_AU45_REGRESSOR    "/face/au/45/regressor"    // Blink
#define OSC_ADDR_FACE_AU45_CLASSIFIER   "/face/au/45/classifier"   // Blink
#define OSC_ADDR_FACE_ATTENTION         "/face/attention"

#define OSC_ADDR_FACE_EMOTION_BOOL_HAPPY     "/face/emotion/bool/happy"
#define OSC_ADDR_FACE_EMOTION_BOOL_SAD       "/face/emotion/bool/sad"
#define OSC_ADDR_FACE_EMOTION_BOOL_SURPRISED "/face/emotion/bool/surprised"
#define OSC_ADDR_FACE_EMOTION_BOOL_DISGUSTED "/face/emotion/bool/disgusted"
#define OSC_ADDR_FACE_EMOTION_BOOL_ANGRY     "/face/emotion/bool/angry"
#define OSC_ADDR_FACE_EMOTION_BOOL_FEARED    "/face/emotion/bool/feared"
#define OSC_ADDR_FACE_EMOTION_BOOL_NEUTRAL   "/face/emotion/bool/neutral"

#define OSC_ADDR_FACE_EMOTION_PROB_HAPPY     "/face/emotion/prob/happy"
#define OSC_ADDR_FACE_EMOTION_PROB_SAD       "/face/emotion/prob/sad"
#define OSC_ADDR_FACE_EMOTION_PROB_SURPRISED "/face/emotion/prob/surprised"
#define OSC_ADDR_FACE_EMOTION_PROB_DISGUSTED "/face/emotion/prob/disgusted"
#define OSC_ADDR_FACE_EMOTION_PROB_ANGRY     "/face/emotion/prob/angry"
#define OSC_ADDR_FACE_EMOTION_PROB_FEARED    "/face/emotion/prob/feared"
#define OSC_ADDR_FACE_EMOTION_PROB_NEUTRAL   "/face/emotion/prob/neutral"

class OscSender
{
protected:
	char                        mBuffer[OSC_OUTPUT_BUFFER_SIZE];
	::UdpTransmitSocket         mSocket;
	::osc::OutboundPacketStream mStream;

public:
	OscSender(
		const char* address = OSC_DEFAULT_ADDRESS, 
		const int   port    = OSC_DEFAULT_PORT) :
		mSocket(IpEndpointName(address, port)),
		mStream(::osc::OutboundPacketStream(mBuffer, OSC_OUTPUT_BUFFER_SIZE))
	{ }

	void send(::OpenFace::Result& item)
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// BASIC

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_DETECTED) << (float)item.faceDetected << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_CERTAINTY) << (float)item.certainty << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// POSITION

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_POSITION_X) << (float)item.position[0] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_POSITION_Y) << (float)item.position[1] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_POSITION_Z) << (float)item.position[2] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// ORIENTATION

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_ORIENTATION_AX) << (float)item.orientation[0] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_ORIENTATION_AY) << (float)item.orientation[1] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_ORIENTATION_AZ) << (float)item.orientation[2] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// EYE-GAZE LEFT

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EYEGAZE_LEFT_X) << (float)item.eyeLeft.gaze[0] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EYEGAZE_LEFT_Y) << (float)item.eyeLeft.gaze[1] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EYEGAZE_LEFT_Z) << (float)item.eyeLeft.gaze[2] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// EYE-GAZE RIGHT

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EYEGAZE_RIGHT_X) << (float)item.eyeRight.gaze[0] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EYEGAZE_RIGHT_Y) << (float)item.eyeRight.gaze[1] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EYEGAZE_RIGHT_Z) << (float)item.eyeRight.gaze[2] << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		/////////////////////////////////////////////////////////////////////////////////////////

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_AU45_REGRESSOR) << (float)item.auSVR.AU45 << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_AU45_CLASSIFIER) << (float)item.auSVM.AU45 << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_ATTENTION) << (float)item.getAverageAttention() << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// EMOTION BOOLS

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_BOOL_HAPPY) << (float)item.emotions.happy << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_BOOL_SAD) << (float)item.emotions.sad << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_BOOL_SURPRISED) << (float)item.emotions.surprised << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_BOOL_DISGUSTED) << (float)item.emotions.disgusted << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_BOOL_ANGRY) << (float)item.emotions.angry << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_BOOL_FEARED) << (float)item.emotions.feared << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_BOOL_NEUTRAL) << (float)item.emotions.neutral << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// EMOTION PROBABILITIES

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_PROB_HAPPY) << (float)item.emotionProbability.happy << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_PROB_SAD) << (float)item.emotionProbability.sad << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_PROB_SURPRISED) << (float)item.emotionProbability.surprised << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_PROB_DISGUSTED) << (float)item.emotionProbability.disgusted << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_PROB_ANGRY) << (float)item.emotionProbability.angry << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_PROB_FEARED) << (float)item.emotionProbability.feared << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());

		mStream.Clear();
		mStream << ::osc::BeginMessage(OSC_ADDR_FACE_EMOTION_PROB_NEUTRAL) << (float)item.emotionProbability.neutral << ::osc::EndMessage;
		mSocket.Send(mStream.Data(), mStream.Size());
	}
};