#pragma once

#ifndef __FACE_RESULT_H
#define __FACE_RESULT_H

#include <opencv2/core.hpp>
#include <FaceActionUnit.h>

namespace OpenFace
{

struct EmotionBools
{
	bool happy     = false;
	bool sad       = false;
	bool disgusted = false;
	bool surprised = false;
	bool feared    = false;
	bool angry     = false;
	bool neutral   = false;
};

struct EmotionProbability
{
	double happy     = 0.0;
	double sad       = 0.0;
	double disgusted = 0.0;
	double surprised = 0.0;
	double feared    = 0.0;
	double angry     = 0.0;
	double neutral   = 0.0;
};

struct Eye
{
	double    attention = 0.0;
	cv::Vec3d position  = ::cv::Vec3d(0.0, 0.0, 0.0);
	cv::Vec3d gaze      = ::cv::Vec3d(0.0, 0.0, 0.0);
};

class Result
{
public:
	bool               faceDetected;
	double             certainty;
	double             modelLikelihood;
	cv::Vec3d          position;
	cv::Vec3d          orientation;
	Eye                eyeLeft;
	Eye                eyeRight;
	ActionUnitValues   auSVR;
	ActionUnitValues   auSVM;
	cv::Mat*           image;
	EmotionBools       emotions;
	EmotionProbability emotionProbability;

	Result() :
		faceDetected(false),
		certainty(0.0),
		modelLikelihood(0.0)
	{ }

	__forceinline double const getAverageAttention() const { return (eyeLeft.attention + eyeRight.attention) * 0.5; }
};
}
#endif
