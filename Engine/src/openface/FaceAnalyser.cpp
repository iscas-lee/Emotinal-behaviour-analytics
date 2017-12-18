///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"

using namespace OpenFace;
using namespace std;

// Constructor from a model file (or a default one if not provided
FaceAnalyser::FaceAnalyser(
	const vector<pair<string, ActionUnit>>& auFiles,
	const string&                           triLocation,
	const vector<pair<string, Emotion>>&    emotionSVRFiles,
	vector<cv::Vec3d>                       orientation_bins, 
	double                                  scale, 
	int                                     width, 
	int                                     height) :
	mAlignScale(scale),
	mAlignWidth(width),
	mAlignHeight(height),
	mNumBinsHog(1000),    // Initialise the histograms that will represent bins from 0 - 1 (as HoG values are only stored as those)
	mMaxValHog(1),
	mMinValHog(-0.005),
	mNumBinsGeom(10000),  // The geometry histogram ranges from -60 to 60
	mMaxValGeom(60),
	mMinValGeom(-60),
	mFramesTracking(0),
	mGeomHistSum(0)
{
	// read in AU files
	ReadAU(auFiles);
		
	if (orientation_bins.empty())
	{
		// Just using frontal currently
		mHeadOrientations.push_back(cv::Vec3d(0,0,0));
	}
	else
	{
		mHeadOrientations = orientation_bins;
	}

	mHogHistSum.resize(mHeadOrientations.size());
	mHogDescHist.resize(mHeadOrientations.size());

	// The triangulation used for masking out the non-face parts of aligned image
	std::ifstream triangulation_file(triLocation);
	ReadMat(triangulation_file, mTriangulation);

	// Load AU to emotion libsvm classifiers
	for (size_t i = 0; i < emotionSVRFiles.size(); i++)
	{
		const pair<string, Emotion>& entry = emotionSVRFiles.at(i);

		switch (entry.second)
		{
		case Emotion::Happy:     mModelHappy    = svm_load_model(entry.first.c_str());
		case Emotion::Sad:       mModelSad      = svm_load_model(entry.first.c_str());
		case Emotion::Surprised: mModelSurprise = svm_load_model(entry.first.c_str());
		case Emotion::Disgusted: mModelDisgust  = svm_load_model(entry.first.c_str());
		case Emotion::Angry:     mModelAngry    = svm_load_model(entry.first.c_str());
		case Emotion::Feared:    mModelFear     = svm_load_model(entry.first.c_str());
		case Emotion::Neutral:   mModelNeutral  = svm_load_model(entry.first.c_str());
		}
	}
}

// Getting the closest view center based on orientation
int GetViewId(const vector<cv::Vec3d> orientations_all, const cv::Vec3d& orientation)
{
	int id = 0;
	double dbest = -1.0;

	for(size_t i = 0; i < orientations_all.size(); i++)
	{
		// Distance to current view
		double d = cv::norm(orientation, orientations_all[i]);

		if(i == 0 || d < dbest)
		{
			dbest = d;
			id = i;
		}
	}

	return id;	
}

void FaceAnalyser::processFrame(const cv::Mat& frame, const CLNF& clnf_model, Result& result, bool visualise)
{
	mFramesTracking++;

	// First align the face if tracking was successfull
	if (clnf_model.isDetectionSuccess())
	{
		AlignFaceMask(mAlignedFace, frame, clnf_model, mTriangulation, true, mAlignScale, mAlignWidth, mAlignHeight);
	}
	else
	{
		mAlignedFace = cv::Mat(mAlignHeight, mAlignWidth, CV_8UC3);
		mAlignedFace.setTo(0);
	}

	if (mAlignedFace.channels() == 3)
	{
		cv::cvtColor(mAlignedFace, mAlignedFaceGrayscale, CV_BGR2GRAY);
	}
	else
	{
		mAlignedFaceGrayscale = mAlignedFace.clone();
	}

	// Extract HOG descriptor from the frame and convert it to a useable format
	cv::Mat_<double> hog_descriptor;
	Extract_FHOG_descriptor(hog_descriptor, mAlignedFace, this->mNumHogRows, this->mNumHogCols);
	
	// Store the descriptor
	mHogDescFrame = hog_descriptor;

	int orientation_to_use = GetViewId(mHeadOrientations, clnf_model.getParamsGlobal().orient);

	// Only update the running median if predictions are not high
	// That is don't update it when the face is expressive (just retrieve it)

	// A small speedup
	if (mFramesTracking % 2 == 1)
	{
		UpdateRunningMedian(
			mHogDescHist[orientation_to_use], 
			mHogHistSum[orientation_to_use],
			mHogDescMedian, 
			hog_descriptor, 
			clnf_model.isDetectionSuccess(),
			mNumBinsHog, 
			mMinValHog,
			mMaxValHog);
		
		mHogDescMedian.setTo(0, mHogDescMedian < 0);
	}	

	// Geom descriptor and its median
	mGeomDescriptorFrame = clnf_model.getParamsLocal().t();
	
	if (!clnf_model.isDetectionSuccess())
		mGeomDescriptorFrame.setTo(0);

	// Stack with the actual feature point locations (without mean)
	cv::Mat_<double> locs = clnf_model.getPDM().getPrincipleComponents() * mGeomDescriptorFrame.t();
	
	cv::hconcat(locs.t(), mGeomDescriptorFrame.clone(), mGeomDescriptorFrame);
	
	// A small speedup
	if (mFramesTracking % 2 == 1)
	{
		UpdateRunningMedian(
			mGeomDescHist,
			mGeomHistSum,
			mGeomDescriptorMedian,
			mGeomDescriptorFrame, 
			clnf_model.isDetectionSuccess(),
			mNumBinsGeom,
			mMinValGeom,
			mMaxValGeom);
	}

	// First convert the face image to double representation as a row vector
	cv::Mat_<uchar> aligned_face_cols(1, mAlignedFace.cols * mAlignedFace.rows * mAlignedFace.channels(), mAlignedFace.data, 1);
	cv::Mat_<double> aligned_face_cols_double;
	aligned_face_cols.convertTo(aligned_face_cols_double, CV_64F);

	// Visualising the median HOG
	if (visualise)	
		Visualise_FHOG(hog_descriptor, mNumHogRows, mNumHogCols, mHogDescriptorVisualisation);
	
	//////////////////////////////////////////////////
	// Perform AU prediction
	if (!mHogDescFrame.empty())
	{
		// SVR
		mAUSVRStaticLinearRegressors.Predict(result.auSVR, mHogDescFrame, mGeomDescriptorFrame);
		mAUSVRDynamicLinearRegressors.Predict(result.auSVR, mHogDescFrame, mGeomDescriptorFrame, mHogDescMedian, mGeomDescriptorMedian);

		// SVM
		mAUSVMStaticLinearClassifiers.Predict(result.auSVM, mHogDescFrame, mGeomDescriptorFrame);
		mAUSVMDynamicLinearClassifiers.Predict(result.auSVM, mHogDescFrame, mGeomDescriptorFrame, mHogDescMedian, mGeomDescriptorMedian);

		// predict emotion booleans from SVM
		predictSingleEmotionFromSVM(result);

		// predict emtion probability from LIBSVM SVR
		svm_node nodes[17+1];
		nodes[0].index = 1;
		nodes[0].value = result.auSVR.AU01;
		nodes[1].index = 2;
		nodes[1].value = result.auSVR.AU02;
		nodes[2].index = 3;
		nodes[2].value = result.auSVR.AU04;
		nodes[3].index = 4;
		nodes[3].value = result.auSVR.AU05;
		nodes[4].index = 5;
		nodes[4].value = result.auSVR.AU06;
		nodes[5].index = 6;
		nodes[5].value = result.auSVR.AU07;
		nodes[6].index = 7;
		nodes[6].value = result.auSVR.AU09;
		nodes[7].index = 8;
		nodes[7].value = result.auSVR.AU10;
		nodes[8].index = 9;
		nodes[8].value = result.auSVR.AU12;
		nodes[9].index = 10;
		nodes[9].value = result.auSVR.AU14;
		nodes[10].index = 11;
		nodes[10].value = result.auSVR.AU15;
		nodes[11].index = 12;
		nodes[11].value = result.auSVR.AU17;
		nodes[12].index = 13;
		nodes[12].value = result.auSVR.AU20;
		nodes[13].index = 14;
		nodes[13].value = result.auSVR.AU23;
		nodes[14].index = 15;
		nodes[14].value = result.auSVR.AU25;
		nodes[15].index = 16;
		nodes[15].value = result.auSVR.AU26;
		nodes[16].index = 17;
		nodes[16].value = result.auSVR.AU45;
		nodes[17].index = -1;

		// libsvm prediction
		result.emotionProbability.happy     = max(0.0, min(svm_predict(mModelHappy, nodes),    1.0));
		result.emotionProbability.sad       = max(0.0, min(svm_predict(mModelSad, nodes),      1.0));
		result.emotionProbability.surprised = max(0.0, min(svm_predict(mModelSurprise, nodes), 1.0));
		result.emotionProbability.disgusted = max(0.0, min(svm_predict(mModelDisgust, nodes),  1.0));
		result.emotionProbability.angry     = max(0.0, min(svm_predict(mModelAngry, nodes),    1.0));
		result.emotionProbability.feared    = max(0.0, min(svm_predict(mModelFear, nodes),     1.0));
		result.emotionProbability.neutral   = max(0.0, min(svm_predict(mModelNeutral, nodes),  1.0));
		//printf("HAPPY: %f\tSAD: %f\tSURPRISE: %f\tDISGUST: %f\n", valu1, valu2, valu3, valu4);
	}
}

// Reset the models
void FaceAnalyser::Reset()
{
	mFramesTracking = 0;
	mHogDescMedian.setTo(cv::Scalar(0.0));
	mFaceImageMedian.setTo(cv::Scalar(0.0));

	for (size_t i = 0; i < mHogDescHist.size(); ++i)
	{
		mHogDescHist[i] = cv::Mat_<unsigned int>(mHogDescHist[i].rows, mHogDescHist[i].cols, (unsigned int)0);
		mHogHistSum[i]  = 0;
	}

	mGeomDescriptorMedian.setTo(cv::Scalar(0.0));
	mGeomDescHist = cv::Mat_<unsigned int>(mGeomDescHist.rows, mGeomDescHist.cols, (unsigned int)0);
	mGeomHistSum = 0;
}

void FaceAnalyser::UpdateRunningMedian(
	cv::Mat_<unsigned int>& histogram, 
	int& hist_count, 
	cv::Mat_<double>& median, 
	const cv::Mat_<double>& descriptor, 
	bool update, 
	int num_bins, 
	double min_val, 
	double max_val)
{
	double length = max_val - min_val;
	if(length < 0)
		length = -length;

	// The median update
	if(histogram.empty())
	{
		histogram = cv::Mat_<unsigned int>(descriptor.cols, num_bins, (unsigned int)0);
		median = descriptor.clone();
	}

	if(update)
	{
		// Find the bins corresponding to the current descriptor
		cv::Mat_<double> converted_descriptor = (descriptor - min_val)*((double)num_bins)/(length);

		// Capping the top and bottom values
		converted_descriptor.setTo(cv::Scalar(num_bins-1), converted_descriptor > num_bins - 1);
		converted_descriptor.setTo(cv::Scalar(0), converted_descriptor < 0);

		for(int i = 0; i < histogram.rows; ++i)
		{
			int index = (int)converted_descriptor.at<double>(i);
			histogram.at<unsigned int>(i, index)++;
		}

		// Update the histogram count
		hist_count++;
	}

	if(hist_count == 1)
	{
		median = descriptor.clone();
	}
	else
	{
		// Recompute the median
		int cutoff_point = (hist_count + 1)/2;

		// For each dimension
		for(int i = 0; i < histogram.rows; ++i)
		{
			int cummulative_sum = 0;
			for(int j = 0; j < histogram.cols; ++j)
			{
				cummulative_sum += histogram.at<unsigned int>(i, j);
				if(cummulative_sum >= cutoff_point)
				{
					median.at<double>(i) = min_val + ((double)j) * (length/((double)num_bins)) + (0.5*(length)/ ((double)num_bins));
					break;
				}
			}
		}
	}
}

// Reading in AU prediction modules
void FaceAnalyser::ReadAU(const vector<pair<string, ActionUnit>>& auFiles)
{
	for (size_t i = 0; i < auFiles.size(); i++)
	{
		const std::pair<std::string, ActionUnit>& entry = auFiles.at(i);

		// read the AU SVR/SVM DAT file
		ifstream streamDAT(entry.first, ios::in | ios::binary);
		if (streamDAT.is_open())
		{
			// First read the input type
			int regressor_type;
			streamDAT.read((char*)&regressor_type, 4);

			// Then parse regressor or classifier
			switch (regressor_type)
			{
			case SVR_LINEAR_STATIC:
				mAUSVRStaticLinearRegressors.Read(streamDAT, entry.second);
				break;

			case SVR_LINEAR_DYNAMIC:
				mAUSVRDynamicLinearRegressors.Read(streamDAT, entry.second);
				break;

			case SVM_LINEAR_STATIC:
				mAUSVMStaticLinearClassifiers.Read(streamDAT, entry.second);
				break;

			case SVM_LINEAR_DYNAMIC:
				mAUSVMDynamicLinearClassifiers.Read(streamDAT, entry.second);
				break;
			}

			streamDAT.close();
		}
		else
		{
			cout << "Couldn't open the AU module file at: " << entry.first << " aborting" << endl;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Pick only the more stable/rigid points under changes of expression
void extract_rigid_points(cv::Mat_<double>& source_points, cv::Mat_<double>& destination_points)
{
	if (source_points.rows == 68)
	{
		cv::Mat_<double> tmp_source = source_points.clone();
		source_points = cv::Mat_<double>();

		// Push back the rigid points (some face outline, eyes, and nose)
		source_points.push_back(tmp_source.row(1));
		source_points.push_back(tmp_source.row(2));
		source_points.push_back(tmp_source.row(3));
		source_points.push_back(tmp_source.row(4));
		source_points.push_back(tmp_source.row(12));
		source_points.push_back(tmp_source.row(13));
		source_points.push_back(tmp_source.row(14));
		source_points.push_back(tmp_source.row(15));
		source_points.push_back(tmp_source.row(27));
		source_points.push_back(tmp_source.row(28));
		source_points.push_back(tmp_source.row(29));
		source_points.push_back(tmp_source.row(31));
		source_points.push_back(tmp_source.row(32));
		source_points.push_back(tmp_source.row(33));
		source_points.push_back(tmp_source.row(34));
		source_points.push_back(tmp_source.row(35));
		source_points.push_back(tmp_source.row(36));
		source_points.push_back(tmp_source.row(39));
		source_points.push_back(tmp_source.row(40));
		source_points.push_back(tmp_source.row(41));
		source_points.push_back(tmp_source.row(42));
		source_points.push_back(tmp_source.row(45));
		source_points.push_back(tmp_source.row(46));
		source_points.push_back(tmp_source.row(47));

		cv::Mat_<double> tmp_dest = destination_points.clone();
		destination_points = cv::Mat_<double>();

		// Push back the rigid points
		destination_points.push_back(tmp_dest.row(1));
		destination_points.push_back(tmp_dest.row(2));
		destination_points.push_back(tmp_dest.row(3));
		destination_points.push_back(tmp_dest.row(4));
		destination_points.push_back(tmp_dest.row(12));
		destination_points.push_back(tmp_dest.row(13));
		destination_points.push_back(tmp_dest.row(14));
		destination_points.push_back(tmp_dest.row(15));
		destination_points.push_back(tmp_dest.row(27));
		destination_points.push_back(tmp_dest.row(28));
		destination_points.push_back(tmp_dest.row(29));
		destination_points.push_back(tmp_dest.row(31));
		destination_points.push_back(tmp_dest.row(32));
		destination_points.push_back(tmp_dest.row(33));
		destination_points.push_back(tmp_dest.row(34));
		destination_points.push_back(tmp_dest.row(35));
		destination_points.push_back(tmp_dest.row(36));
		destination_points.push_back(tmp_dest.row(39));
		destination_points.push_back(tmp_dest.row(40));
		destination_points.push_back(tmp_dest.row(41));
		destination_points.push_back(tmp_dest.row(42));
		destination_points.push_back(tmp_dest.row(45));
		destination_points.push_back(tmp_dest.row(46));
		destination_points.push_back(tmp_dest.row(47));
	}
}

// Aligning a face to a common reference frame
void FaceAnalyser::AlignFaceMask(cv::Mat& aligned_face, const cv::Mat& frame, const CLNF& clnf_model, const cv::Mat_<int>& triangulation, bool rigid, double sim_scale, int out_width, int out_height)
{
	// Will warp to scaled mean shape
	cv::Mat_<double> similarity_normalised_shape = clnf_model.getPDM().getMeanShape() * sim_scale;

	// Discard the z component
	similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2 * similarity_normalised_shape.rows / 3)).clone();

	cv::Mat_<double> source_landmarks = clnf_model.getDetectedLandmarks().reshape(1, 2).t();
	cv::Mat_<double> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();

	// Aligning only the more rigid points
	if (rigid)	
		extract_rigid_points(source_landmarks, destination_landmarks);
	
	cv::Matx22d scale_rot_matrix = AlignShapesWithScale(source_landmarks, destination_landmarks);
	cv::Matx23d warp_matrix;

	warp_matrix(0, 0) = scale_rot_matrix(0, 0);
	warp_matrix(0, 1) = scale_rot_matrix(0, 1);
	warp_matrix(1, 0) = scale_rot_matrix(1, 0);
	warp_matrix(1, 1) = scale_rot_matrix(1, 1);

	const double tx = clnf_model.getParamsGlobal().tx;
	const double ty = clnf_model.getParamsGlobal().ty;

	cv::Vec2d T(tx, ty);
	T = scale_rot_matrix * T;

	// Make sure centering is correct
	warp_matrix(0, 2) = -T(0) + out_width / 2;
	warp_matrix(1, 2) = -T(1) + out_height / 2;

	cv::warpAffine(frame, aligned_face, warp_matrix, cv::Size(out_width, out_height), cv::INTER_LINEAR);

	// Move the destination landmarks there as well
	cv::Matx22d warp_matrix_2d(warp_matrix(0, 0), warp_matrix(0, 1), warp_matrix(1, 0), warp_matrix(1, 1));

	destination_landmarks = cv::Mat(clnf_model.getDetectedLandmarks().reshape(1, 2).t()) * cv::Mat(warp_matrix_2d).t();

	destination_landmarks.col(0) = destination_landmarks.col(0) + warp_matrix(0, 2);
	destination_landmarks.col(1) = destination_landmarks.col(1) + warp_matrix(1, 2);

	// Move the eyebrows up to include more of upper face
	destination_landmarks.at<double>(0, 1) -= 30;
	destination_landmarks.at<double>(16, 1) -= 30;

	destination_landmarks.at<double>(17, 1) -= 30;
	destination_landmarks.at<double>(18, 1) -= 30;
	destination_landmarks.at<double>(19, 1) -= 30;
	destination_landmarks.at<double>(20, 1) -= 30;
	destination_landmarks.at<double>(21, 1) -= 30;
	destination_landmarks.at<double>(22, 1) -= 30;
	destination_landmarks.at<double>(23, 1) -= 30;
	destination_landmarks.at<double>(24, 1) -= 30;
	destination_landmarks.at<double>(25, 1) -= 30;
	destination_landmarks.at<double>(26, 1) -= 30;

	destination_landmarks = cv::Mat(destination_landmarks.t()).reshape(1, 1).t();

	PAW paw(destination_landmarks, triangulation, 0, 0, aligned_face.cols - 1, aligned_face.rows - 1);

	// Mask each of the channels (a bit of a roundabout way, but OpenCV 3.1 in debug mode doesn't seem to be able to handle a more direct way using split and merge)
	vector<cv::Mat> aligned_face_channels(aligned_face.channels());

	for (int c = 0; c < aligned_face.channels(); ++c)	
		cv::extractChannel(aligned_face, aligned_face_channels[c], c);
	
	for (size_t i = 0; i < aligned_face_channels.size(); ++i)	
		cv::multiply(aligned_face_channels[i], paw.getPixelMask(), aligned_face_channels[i], 1.0, CV_8U);
	
	if (aligned_face.channels() == 3)
	{
		cv::Mat planes[] = { aligned_face_channels[0], aligned_face_channels[1], aligned_face_channels[2] };
		cv::merge(planes, 3, aligned_face);
	}
	else
	{
		aligned_face = aligned_face_channels[0];
	}
}

void FaceAnalyser::Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation)
{
	// First convert to dlib format
	dlib::array2d<dlib::matrix<float, 31, 1>> hog(num_rows, num_cols);

	cv::MatConstIterator_<double> descriptor_it = descriptor.begin();

	for (int y = 0; y < num_cols; ++y)	
		for (int x = 0; x < num_rows; ++x)		
			for (unsigned int o = 0; o < 31; ++o)			
				hog[y][x](o) = *descriptor_it++;
			
	// Draw the FHOG to OpenCV format
	auto fhog_vis = dlib::draw_fhog(hog);
	visualisation = dlib::toMat(fhog_vis).clone();
}

// Create a row vector Felzenszwalb HOG descriptor from a given image
void FaceAnalyser::Extract_FHOG_descriptor(cv::Mat_<double>& descriptor, const cv::Mat& image, int& num_rows, int& num_cols, int cell_size)
{
	dlib::array2d<dlib::matrix<float, 31, 1>> hog;

	if (image.channels() == 1)
	{
		dlib::cv_image<uchar> dlib_warped_img(image);
		dlib::extract_fhog_features(dlib_warped_img, hog, cell_size);
	}
	else
	{
		dlib::cv_image<dlib::bgr_pixel> dlib_warped_img(image);
		dlib::extract_fhog_features(dlib_warped_img, hog, cell_size);
	}

	// Convert to a usable format
	num_cols = hog.nc();
	num_rows = hog.nr();

	descriptor = cv::Mat_<double>(1, num_cols * num_rows * 31);
	cv::MatIterator_<double> descriptor_it = descriptor.begin();
	
	for (int y = 0; y < num_cols; ++y)
		for (int x = 0; x < num_rows; ++x)
			for (unsigned int o = 0; o < 31; ++o)	
				*descriptor_it++ = (double)hog[y][x](o);
}

void FaceAnalyser::predictSingleEmotionFromSVM(Result& result)
{
	// map 1.0 double to true
	const bool AU01 = result.auSVM.AU01 == 1.0;
	const bool AU02 = result.auSVM.AU02 == 1.0;
	const bool AU04 = result.auSVM.AU04 == 1.0;
	const bool AU05 = result.auSVM.AU05 == 1.0;
	const bool AU06 = result.auSVM.AU06 == 1.0;
	const bool AU07 = result.auSVM.AU07 == 1.0;
	const bool AU09 = result.auSVM.AU09 == 1.0;
	const bool AU10 = result.auSVM.AU10 == 1.0;
	const bool AU12 = result.auSVM.AU12 == 1.0;
	const bool AU14 = result.auSVM.AU14 == 1.0;
	const bool AU15 = result.auSVM.AU15 == 1.0;
	const bool AU17 = result.auSVM.AU17 == 1.0;
	const bool AU20 = result.auSVM.AU20 == 1.0;
	const bool AU23 = result.auSVM.AU23 == 1.0;
	const bool AU25 = result.auSVM.AU25 == 1.0;
	const bool AU26 = result.auSVM.AU26 == 1.0;
	const bool AU45 = result.auSVM.AU45 == 1.0;

	// expressions for emotions from AU
	const bool HAPPY = AU06 && AU12;
	const bool SAD   = AU15 && !AU06 && !AU12 && !(AU25 && AU26);
	const bool DISGU = AU09 && AU10 && !HAPPY;
	const bool SURPR = AU01 && AU02 && AU05 && AU25;
	const bool FEAR  = AU01 && AU02 && AU05 && (AU07 || AU20) && AU25 && !(AU12 && AU06);
	const bool ANGRY = AU04 && AU07 && AU23 && !(AU12 && AU06);
	
	// must have two true in a row to actually switch to true (noise filter)
	result.emotions.happy     = mLastEmotionBools.happy     && HAPPY;
	result.emotions.sad       = mLastEmotionBools.sad       && SAD;
	result.emotions.disgusted = mLastEmotionBools.disgusted && DISGU;
	result.emotions.surprised = mLastEmotionBools.surprised && SURPR;
	result.emotions.feared    = mLastEmotionBools.feared    && FEAR;
	result.emotions.angry     = mLastEmotionBools.angry     && ANGRY;

	// neutral is true if no other is
	result.emotions.neutral = !(
		result.emotions.happy     | result.emotions.sad    | result.emotions.disgusted |
		result.emotions.surprised | result.emotions.feared | result.emotions.angry);

	// remember as last
	mLastEmotionBools.happy     = HAPPY;
	mLastEmotionBools.sad       = SAD;
	mLastEmotionBools.disgusted = DISGU;
	mLastEmotionBools.surprised = SURPR;
	mLastEmotionBools.feared    = FEAR;
	mLastEmotionBools.angry     = ANGRY;

	//printf("HA:%i SA:%i DI:%i SU:%i FE:%i AN:%i \n", HAPPY, SAD, DISGU, SURPR, FEAR, ANGRY);
}
