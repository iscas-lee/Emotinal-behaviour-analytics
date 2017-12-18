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

#ifndef __FACE_ANALYSER_h_
#define __FACE_ANALYSER_h_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <libsvm/svm.h>

#include <FaceActionUnit.h>
#include <FaceModel.h>
#include <FaceResult.h>

namespace OpenFace
{

enum Emotion
{
	Happy, Sad, Surprised, Disgusted, Angry, Feared, Neutral
};

class FaceAnalyser
{
public:

	enum RegressorType 
	{ 
		SVR_LINEAR_STATIC  = 0, 
		SVR_LINEAR_DYNAMIC = 1, 
		SVM_LINEAR_STATIC  = 4, 
		SVM_LINEAR_DYNAMIC = 5, 
	};

	// Constructor from a model file (or a default one if not provided
	// TODO scale width and height should be read in as part of the model as opposed to being here?
	FaceAnalyser(
		const vector<pair<string, ActionUnit>>& auFiles,
		const string&                           triLocation,
		const vector<pair<string, Emotion>>&    emotionSVRFiles,
		vector<cv::Vec3d>                       orientation_bins = vector<cv::Vec3d>(), 
		double                                  scale            = 0.7, 
		int                                     width            = 112, 
		int                                     height           = 112);

	void processFrame(const cv::Mat& frame, const CLNF& clnf, Result& result, bool visualise = true);

	__forceinline void GetLatestAlignedFace(cv::Mat& image)           const { image     = mAlignedFace.clone();          }
	__forceinline cv::Mat GetLatestHOGDescriptorVisualisation()       const { return mHogDescriptorVisualisation; }
	__forceinline cv::Mat_<uchar> GetLatestAlignedFaceGrayscale()     const { return mAlignedFaceGrayscale.clone(); }
	
	void Reset();

protected:
	int                            mFramesTracking;                // counter for processed frames
	SVRStaticLinear                mAUSVRStaticLinearRegressors;   // The linear SVR static regressors
	SVRDynamicLinear               mAUSVRDynamicLinearRegressors;  // The linear SVR dynamic regressors
	SVMStaticLinear                mAUSVMStaticLinearClassifiers;  // The linear SVM static classifiers
	SVMDynamicLinear               mAUSVMDynamicLinearClassifiers; // The linear SVM dynamic classifiers
	cv::Mat_<uchar>                mAlignedFaceGrayscale;          // Cached intermediate image
	cv::Mat                        mAlignedFace;                   // Cached intermediate image
	cv::Mat                        mHogDescriptorVisualisation;    // Cached intermediate image
	cv::Mat_<double>               mHogDescFrame;                  // The HOG descriptor of the last frame
	int                            mNumHogRows;
	int                            mNumHogCols;
	cv::Mat_<double>               mHogDescMedian;                 // Keep a running median of the hog descriptors and a aligned images
	cv::Mat_<double>               mFaceImageMedian;               // Keep a running median of the hog descriptors and a aligned images	
	vector<cv::Mat_<unsigned int>> mHogDescHist;                   // Use histograms for quick (but approximate) median computation
	vector<cv::Vec3d>              mHeadOrientations;
	int                            mNumBinsHog;
	double                         mMinValHog;
	double                         mMaxValHog;
	vector<int>                    mHogHistSum;
	cv::Mat_<double>               mGeomDescriptorFrame;           // The geometry descriptor (rigid followed by non-rigid shape parameters from CLNF)
	cv::Mat_<double>               mGeomDescriptorMedian;
	cv::Mat_<unsigned int>         mGeomDescHist;
	int                            mGeomHistSum;
	int                            mNumBinsGeom;
	double                         mMinValGeom;
	double                         mMaxValGeom;
	cv::Mat_<int>                  mTriangulation;                 // Used for face alignment
	double                         mAlignScale;                    // Used for face alignment
	int                            mAlignWidth;                    // Used for face alignment
	int                            mAlignHeight;                   // Used for face alignment
	EmotionBools                   mLastEmotionBools;
	svm_model*                     mModelHappy;                    // libsvm classifier AU->Emotion
	svm_model*                     mModelSad;                      // libsvm classifier AU->Emotion
	svm_model*                     mModelSurprise;                 // libsvm classifier AU->Emotion
	svm_model*                     mModelDisgust;                  // libsvm classifier AU->Emotion
	svm_model*                     mModelAngry;                    // libsvm classifier AU->Emotion
	svm_model*                     mModelFear;                     // libsvm classifier AU->Emotion
	svm_model*                     mModelNeutral;                  // libsvm classifier AU->Emotion

	////////////////////////////////////////////////////

	/// <summary>
	/// Reads the defined AU DAT files from specified folder
	/// </summary>
	void ReadAU(const vector<pair<string, ActionUnit>>& auFiles);

	// A utility function for keeping track of approximate running medians used for AU and emotion inference using a set of histograms (the histograms are evenly spaced from min_val to max_val)
	// Descriptor has to be a row vector
	void UpdateRunningMedian(cv::Mat_<unsigned int>& histogram, int& hist_sum, cv::Mat_<double>& median, const cv::Mat_<double>& descriptor, bool update, int num_bins, double min_val, double max_val);

	// Aligning a face to a common reference frame
	void AlignFaceMask(cv::Mat& aligned_face, const cv::Mat& frame, const CLNF& clnf_model, const cv::Mat_<int>& triangulation, bool rigid = true, double scale = 0.6, int width = 96, int height = 96);
	
	void Extract_FHOG_descriptor(cv::Mat_<double>& descriptor, const cv::Mat& image, int& num_rows, int& num_cols, int cell_size = 8);
	void Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation);

	void predictSingleEmotionFromSVM(Result& result);
};
  //===========================================================================
}
#endif
