#pragma once

#ifndef __FACE_SYSTEM_H
#define __FACE_SYSTEM_H

#include <FaceResult.h>
#include <FaceWorker.h>
#include <FaceModelParameters.h>
#include <FaceDetector.h>
#include <FaceModel.h>
#include <FaceValidator.h>
#include <FaceAnalyser.h>

namespace OpenFace
{
class FaceSystemCommand
{
public:
	enum Type { None, Reset };
	__forceinline virtual const Type getType() const { return Type::None; }
};

/// <summary>
/// Main Class bundling all others and providing API.
/// </summary>
class FaceSystem : public Worker<::cv::Mat*, Result*, FaceSystemCommand*>
{
public:
	class CommandReset : public FaceSystemCommand
	{
	public:
		__forceinline const Type getType() const override { return Type::Reset; }
	};

protected:
	FaceModelParameters mFaceModelParams;
	FaceDetectorDlib    mFaceDetector;
	DetectionValidator  mValidator;
	ModelMain           mCLNF;
	FaceAnalyser        mFaceAnalyser;

	void init()                              override;
	void shutdown()                          override;
	void process(::cv::Mat* item)            override;
	void execute(FaceSystemCommand* command) override;

public:
	/// <summary>
	/// Default Constructor using Resource Path
	/// </summary>
	FaceSystem(const string& resourcePath = "./res/");

	/// <summary>
	/// Constructor with manual resources
	/// </summary>
	FaceSystem(
		const string&                           mainPdmFile,
		const string&                           mainValidatorFile,
		const vector<string>&                   mainCcnfFiles,
		const string&                           innerPdmFile,
		const vector<string>&                   innerCcnfFiles,
		const string&                           leftEyePdmFile,
		const vector<string>&                   leftEyeCcnfFiles,
		const string&                           rightEyePdmFile,
		const vector<string>&                   rightEyeCcnfFiles,
		const string&                           mainTriangulationsFile,
		const vector<pair<string, ActionUnit>>& auFiles,
		const string&                           auTriangulationsFile,
		const vector<pair<string, Emotion>>&    emotionSVRFiles
	);

	inline Result* dequeue() { return mQueueOut.dequeue(); }
	inline void    reset()   { mCommands.enqueue(new CommandReset()); }
};
}
#endif
