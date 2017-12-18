#include "stdafx.h"

using namespace OpenFace;

FaceSystem::FaceSystem(const string& resourcePath) : Worker(),
	mValidator(resourcePath + "validator_general_68.dat"),
	mCLNF(
		resourcePath + "In-the-wild_aligned_PDM_68.txt",
		vector<string> () =
		{
			resourcePath + "ccnf_patches_0.25_general.dat",
			resourcePath + "ccnf_patches_0.35_general.dat",
			resourcePath + "ccnf_patches_0.5_general.dat"
		}, 
		resourcePath + "pdm_51_inner.txt",
		vector<string>() = 
		{
			resourcePath + "ccnf_patches_1.00_inner.dat"
		},
		resourcePath + "pdm_28_l_eye_3D_closed.txt",
		vector<string>() =
		{
			resourcePath + "left_ccnf_patches_1.00_synth_lid_.dat",
			resourcePath + "left_ccnf_patches_1.50_synth_lid_.dat"
		},
		resourcePath + "pdm_28_eye_3D_closed.txt",
		vector<string>() =
		{
			resourcePath + "ccnf_patches_1.00_synth_lid_.dat",
			resourcePath + "ccnf_patches_1.50_synth_lid_.dat"
		},
		resourcePath + "tris_68.txt"),
	mFaceAnalyser(
		vector<pair<string, ActionUnit>>()  =
		{
			pair<string, ActionUnit>(resourcePath + "au_01_svm_stat.dat", ActionUnit::AU01),
			pair<string, ActionUnit>(resourcePath + "au_02_svm_stat.dat", ActionUnit::AU02),
			pair<string, ActionUnit>(resourcePath + "au_04_svm_stat.dat", ActionUnit::AU04),
			pair<string, ActionUnit>(resourcePath + "au_05_svm_stat.dat", ActionUnit::AU05),
			pair<string, ActionUnit>(resourcePath + "au_06_svm_stat.dat", ActionUnit::AU06),
			pair<string, ActionUnit>(resourcePath + "au_07_svm_stat.dat", ActionUnit::AU07),
			pair<string, ActionUnit>(resourcePath + "au_09_svm_stat.dat", ActionUnit::AU09),
			pair<string, ActionUnit>(resourcePath + "au_10_svm_stat.dat", ActionUnit::AU10),
			pair<string, ActionUnit>(resourcePath + "au_12_svm_stat.dat", ActionUnit::AU12),
			pair<string, ActionUnit>(resourcePath + "au_14_svm_stat.dat", ActionUnit::AU14),
			pair<string, ActionUnit>(resourcePath + "au_15_svm_stat.dat", ActionUnit::AU15),
			pair<string, ActionUnit>(resourcePath + "au_17_svm_stat.dat", ActionUnit::AU17),
			pair<string, ActionUnit>(resourcePath + "au_20_svm_stat.dat", ActionUnit::AU20),
			pair<string, ActionUnit>(resourcePath + "au_23_svm_stat.dat", ActionUnit::AU23),
			pair<string, ActionUnit>(resourcePath + "au_25_svm_stat.dat", ActionUnit::AU25),
			pair<string, ActionUnit>(resourcePath + "au_26_svm_stat.dat", ActionUnit::AU26),
			pair<string, ActionUnit>(resourcePath + "au_28_svm_stat.dat", ActionUnit::AU28),
			pair<string, ActionUnit>(resourcePath + "au_45_svm_stat.dat", ActionUnit::AU45),
			pair<string, ActionUnit>(resourcePath + "au_01_svr_stat.dat", ActionUnit::AU01),
			pair<string, ActionUnit>(resourcePath + "au_02_svr_stat.dat", ActionUnit::AU02),
			pair<string, ActionUnit>(resourcePath + "au_04_svr_stat.dat", ActionUnit::AU04),
			pair<string, ActionUnit>(resourcePath + "au_05_svr_stat.dat", ActionUnit::AU05),
			pair<string, ActionUnit>(resourcePath + "au_06_svr_stat.dat", ActionUnit::AU06),
			pair<string, ActionUnit>(resourcePath + "au_07_svr_stat.dat", ActionUnit::AU07),
			pair<string, ActionUnit>(resourcePath + "au_09_svr_stat.dat", ActionUnit::AU09),
			pair<string, ActionUnit>(resourcePath + "au_10_svr_stat.dat", ActionUnit::AU10),
			pair<string, ActionUnit>(resourcePath + "au_12_svr_stat.dat", ActionUnit::AU12),
			pair<string, ActionUnit>(resourcePath + "au_14_svr_stat.dat", ActionUnit::AU14),
			pair<string, ActionUnit>(resourcePath + "au_15_svr_stat.dat", ActionUnit::AU15),
			pair<string, ActionUnit>(resourcePath + "au_17_svr_stat.dat", ActionUnit::AU17),
			pair<string, ActionUnit>(resourcePath + "au_20_svr_stat.dat", ActionUnit::AU20),
			pair<string, ActionUnit>(resourcePath + "au_23_svr_stat.dat", ActionUnit::AU23),
			pair<string, ActionUnit>(resourcePath + "au_25_svr_stat.dat", ActionUnit::AU25),
			pair<string, ActionUnit>(resourcePath + "au_26_svr_stat.dat", ActionUnit::AU26),
			pair<string, ActionUnit>(resourcePath + "au_45_svr_stat.dat", ActionUnit::AU45)
		},
		resourcePath + "tris_68_full.txt",
		vector<pair<string, Emotion>>()  =
		{
			pair<string, Emotion>(resourcePath + "emot.happy.train.model",    Emotion::Happy),
			pair<string, Emotion>(resourcePath + "emot.sad.train.model",      Emotion::Sad),
			pair<string, Emotion>(resourcePath + "emot.surprise.train.model", Emotion::Surprised),
			pair<string, Emotion>(resourcePath + "emot.disgust.train.model",  Emotion::Disgusted),
			pair<string, Emotion>(resourcePath + "emot.angry.train.model",    Emotion::Angry),
			pair<string, Emotion>(resourcePath + "emot.fear.train.model",     Emotion::Feared),
			pair<string, Emotion>(resourcePath + "emot.neutral.train.model",  Emotion::Neutral)
		})
{
	mFaceModelParams.track_gaze = true;
}

FaceSystem::FaceSystem(
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
	const vector<pair<string, Emotion>>&    emotionSVRFiles) : Worker(),
	mValidator(mainValidatorFile),
	mCLNF(mainPdmFile, mainCcnfFiles, innerPdmFile, innerCcnfFiles, leftEyePdmFile, leftEyeCcnfFiles, rightEyePdmFile, rightEyeCcnfFiles, mainTriangulationsFile),
	mFaceAnalyser(auFiles, auTriangulationsFile, emotionSVRFiles)
{
	mFaceModelParams.track_gaze = true;
}

void FaceSystem::init()
{
	Worker::init();

}

void FaceSystem::shutdown()
{
	Worker::shutdown();
}

void FaceSystem::process(::cv::Mat* item)
{
	/////////////////////////////////////////////////////////////////////////////////////////

	Result* result = new Result();
	memset(&result->auSVM, 0, sizeof(result->auSVM));
	memset(&result->auSVR, 0, sizeof(result->auSVR));
	memset(&result->emotions, 0, sizeof(result->emotions));
	::cv::Mat grayFrame;

	//::cv::Vec6d pose;

	// create gray variant
	::cv::cvtColor(*item, grayFrame, CV_RGB2GRAY);

	// some calcs for eye-gaze (TODO: move me out of loop)
	float cx = item->cols / 2.0f;
	float cy = item->rows / 2.0f;
	float fx = 500.0f * ((float)item->cols / 640.0f);
	float fy = 500.0f * ((float)item->rows / 480.0f);
	fx = (fx + fy) / 2.0f;
	fy = fx;

	/////////////////////////////////////////////////////////////////////////////////////////

	bool success = mCLNF.DetectLandmarksInVideo(mFaceDetector, grayFrame, mFaceModelParams, &mValidator);

	if (success)
	{
		mFaceAnalyser.processFrame(*item, mCLNF, *result, true);

		// get pose
		mCLNF.GetCorrectedPoseWorld(result->position, result->orientation, fx, fy, cx, cy);

		// visualize
		mCLNF.Draw(*item);
		mCLNF.DrawBox(*item, result->position, result->orientation, 0xFFFF0000, 1, fx, fy, cx, cy);

		mCLNF.EstimateGaze(*result, fx, fy, cx, cy, true);
		mCLNF.EstimateGaze(*result, fx, fy, cx, cy, false);

		// visualize
		mCLNF.DrawGaze(*item, *result, fx, fy, cx, cy);
	
		//mFaceAnalyser->GetGeomDescriptor(*item);
		//*item = mFaceAnalyser->GetLatestAlignedFaceGrayscale();
		//mFaceAnalyser->GetLatestAlignedFace(*item);
		//*item = mFaceAnalyser->GetLatestHOGDescriptorVisualisation();
	}

	// AU provided:
	// REGR: 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45
	// CLAS: 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45

	/////////////////////////////////////////////////////////////////////////////////////////
	// build Result

	result->faceDetected = mCLNF.isDetectionSuccess();
	result->certainty = 0.5 * (1.0 - mCLNF.getDetectionCertainty());
	result->modelLikelihood = mCLNF.getModelLikelihood();

	/////////////////////////////////////////////////////////////////////////////////////////
	// send result to output queue

	result->image = item;

	if (!mQueueOut.enqueue(result))
	{
		delete item;
		delete result;
	}
}

void FaceSystem::execute(FaceSystemCommand* command)
{
	if (!command)
		return;
	
	switch (command->getType())
	{
	case FaceSystemCommand::Type::Reset:
		mCLNF.Reset();
		mFaceAnalyser.Reset();
		// free queue items
		while (::cv::Mat* item = mQueueIn.dequeue())
			delete item;
		break;
	}

	delete command;
}
