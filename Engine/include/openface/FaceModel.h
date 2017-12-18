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

#ifndef __FACE_MODEL_h_
#define __FACE_MODEL_h_

#include <opencv2/core/core.hpp>
#include "PDM.h"
#include "FacePatchExperts.h"
#include "FaceModelParameters.h"
#include "FaceGlobalParameters.h"
#include "FaceDetector.h"
#include "FaceValidator.h"
#include "FaceResult.h"

using namespace std;

namespace OpenFace
{

// A main class containing all the modules required for landmark detection
// Face shape model
// Patch experts
// Optimization techniques
class CLNF
{
public:
	enum Type { Main, Inner, LeftEye, RightEye };

	__forceinline const bool              isDetectionSuccess()        const { return mDetectionSuccess;       }
	__forceinline const Type              getType()                   const { return mType;                   }
	__forceinline const double            getDetectionCertainty()     const { return mDetectionCertainty;     }
	__forceinline const double            getModelLikelihood()        const { return mModelLikelihood;        }
	__forceinline const PDM&              getPDM()                    const { return mPDM;                    }
	__forceinline const PatchExperts&     getPatchExperts()           const { return mPatchExperts;           }
	__forceinline const cv::Mat_<double>& getDetectedLandmarks()      const { return mDetectedLandmarks;      }
	__forceinline const cv::Mat_<double>& getParamsLocal()            const { return mParamsLocal;            }
	__forceinline const GlobalParameters& getParamsGlobal()           const { return mParamsGlobal;           }
	
	__forceinline cv::Mat_<double>&       getParamsLocal()                  { return mParamsLocal;            }
	__forceinline GlobalParameters&       getParamsGlobal()                 { return mParamsGlobal;           }
	__forceinline PDM&                    getPDM()                          { return mPDM;                    }

	__forceinline void calcShape2D()    { mPDM.CalcShape2D(mDetectedLandmarks, mParamsLocal, mParamsGlobal);  }
	__forceinline void calcParams()     { mPDM.CalcParams(mParamsGlobal, mParamsLocal, mDetectedLandmarks);   }

	
	/// <summary>
	/// Constructor without further sub-models.
	/// </summary>
	CLNF(
		const string&         mainPdmFile,
		const vector<string>& mainCcnfFiles,
		const string&         mainTriangulationsFile = "",
		const Type            modelType = Type::Main);

	// Does the actual work - landmark detection
	bool DetectLandmarks(const cv::Mat_<uchar>& image, const cv::Mat_<float>& depth, FaceModelParameters& params);
	
	// Gets the shape of the current detected landmarks in camera space (given camera calibration)
	// Can only be called after a call to DetectLandmarksInVideo or DetectLandmarksInImage
	cv::Mat_<double> GetShape(double fx, double fy, double cx, double cy) const;

	
	// A utility bounding box function
	cv::Rect_<double> GetBoundingBox() const;

	// Reset the model (useful if we want to completelly reinitialise, or we want to track another video)
	void Reset();

	// Read from files
	void Read(const string& pdmFile, const vector<string>& ccnfFiles, const string& triangulationsFile = "");

	//================================================================
	// Helper function for getting head pose from CLNF parameters

	// Return the current estimate of the head pose, this can be either in camera or world coordinate space
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6d GetPoseCamera(double fx, double fy, double cx, double cy);
	cv::Vec6d GetPoseWorld(double fx, double fy, double cx, double cy);

	// Getting a head pose estimate from the currently detected landmarks, with appropriate correction for perspective
	// This is because rotation estimate under orthographic assumption is only correct close to the centre of the image
	// These methods attempt to correct for that
	// The pose returned can be either in camera or world coordinates
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6d GetCorrectedPoseCamera(double fx, double fy, double cx, double cy);
	void GetCorrectedPoseWorld(cv::Vec3d& position, cv::Vec3d& orientation, double fx, double fy, double cx, double cy);

	//===========================================================================
	// Visualisation functions
	//===========================================================================

	void DrawBox(cv::Mat& image, cv::Vec3d& position, cv::Vec3d& orientation, const cv::Scalar color, const int thickness, const float fx, const float fy, const float cx, const float cy);
	virtual void Draw(cv::Mat& img);


protected:
	Type                           mType;                   // Type of the model (to differ between main/inner/leye/reye)
	PDM	                           mPDM;                    // The linear 3D Point Distribution Model
	PatchExperts                   mPatchExperts;           // The set of patch experts
	int                            mFailuresInARow;         // Keeping track of how many frames the tracker has failed in so far when tracking in videos
	double                         mModelLikelihood;        // The landmark detection likelihoods (combined and per patch expert)
	bool                           mDetectionSuccess;       // Indicating if landmark detection succeeded (based on SVR validator)
	bool                           mTrackingInitialised;    // Indicating if the tracking has been initialised (for video based tracking)
	double                         mDetectionCertainty;     // The actual output of the regressor (-1 is perfect detection 1 is worst detection)
	cv::Mat_<uchar>                mFaceTemplate;           // A template of a face that last succeeded with tracking (useful for large motions in video)
	vector<cv::Mat_<int>>          mTriangulations;         // the triangulation per each view (for drawing purposes only) UNUSED ?????
	cv::Mat_<double>               mDetectedLandmarks;      // Lastly detect 2D model shape [x1,x2,...xn,y1,...yn]
	cv::Mat_<double>               mLandmarkLikelihoods;
	cv::Mat_<double>               mParamsLocal;            // Local parameters describing the non-rigid shape
	GlobalParameters               mParamsGlobal;           // Global parameters describing the rigid shape [scale, euler_x, euler_y, euler_z, tx, ty]
	map<int, cv::Mat_<float>>      mKdeRespPrecalc;         // the speedup of RLMS using precalculated KDE responses (described in Saragih 2011 RLMS paper)

	// The model fitting: patch response computation and optimisation steps
    bool Fit(const cv::Mat_<uchar>& intensity_image, const cv::Mat_<float>& depth_image, const std::vector<int>& window_sizes, const FaceModelParameters& parameters);

	// Mean shift computation that uses precalculated kernel density estimators (the one actually used)
	void NonVectorisedMeanShift_precalc_kde(
		cv::Mat_<float>&               out_mean_shifts, 
		const vector<cv::Mat_<float>>& patch_expert_responses, 
		const cv::Mat_<float>&         dxs, 
		const cv::Mat_<float>&         dys, 
		int                            resp_size, 
		float                          a, 
		int                            scale, 
		int                            view_id, 
		map<int, cv::Mat_<float>>&     mean_shifts);

	// The actual model optimisation (update step), returns the model likelihood
    double NU_RLMS(
		GlobalParameters&              final_global, 
		cv::Mat_<double>&              final_local, 
		const vector<cv::Mat_<float>>& patch_expert_responses, 
		const GlobalParameters&        initial_global, 
		const cv::Mat_<double>&        initial_local,
		const cv::Mat_<double>&        base_shape, 
		const cv::Matx22d&             sim_img_to_ref, 
		const cv::Matx22f&             sim_ref_to_img, 
		int                            resp_size, 
		int                            view_idx, 
		bool                           rigid, 
		int                            scale, 
		cv::Mat_<double>&              landmark_lhoods, 
		const FaceModelParameters&     parameters);

	// Removing background image from the depth
	bool RemoveBackground(cv::Mat_<float>& out_depth_image, const cv::Mat_<float>& depth_image);

	// Generating the weight matrix for the Weighted least squares
	void GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const FaceModelParameters& parameters);

	void UpdateTemplate(const cv::Mat_<uchar> &grayscale_image);
	void CorrectGlobalParametersVideo(const cv::Mat_<uchar> &grayscale_image, const FaceModelParameters& params);


	vector<cv::Point2d> CalculateLandmarks(const cv::Mat_<double>& shape2D, cv::Mat_<int>& visibilities);
	vector<cv::Point2d> CalculateLandmarks(cv::Mat img, const cv::Mat_<double>& shape2D);
	vector<cv::Point2d> CalculateLandmarks();

	void Draw(cv::Mat& img, const cv::Mat_<double>& shape2D, const cv::Mat_<int>& visibilities);
};

//===========================================================================


///
///
///
class ModelMain : public CLNF
{
public:
	/// <summary>
	/// Constructor
	/// </summary>
	ModelMain(
		const string&         mainPdmFile,
		const vector<string>& mainCcnfFiles,
		const string&         innerPdmFile,
		const vector<string>& innerCcnfFiles,
		const string&         leftEyePdmFile,
		const vector<string>& leftEyeCcnfFiles,
		const string&         rightEyePdmFile,
		const vector<string>& rightEyeCcnfFiles,
		const string&         mainTriangulationsFile = "");

	bool DetectLandmarks(const cv::Mat_<uchar>& image, const cv::Mat_<float>& depth, FaceModelParameters& params, DetectionValidator* validator = 0);
	virtual void Draw(cv::Mat& img) override;

	void DrawGaze(cv::Mat& img, Result& result, float fx, float fy, float cx, float cy);
	void EstimateGaze(Result& result, float fx, float fy, float cx, float cy, bool left_eye);

	bool DetectLandmarksInVideo(FaceDetectorDlib& face_detector, const cv::Mat_<uchar> &grayscale_image, FaceModelParameters& params, DetectionValidator* validator = 0);
	bool DetectLandmarksInVideo(FaceDetectorDlib& face_detector, const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, FaceModelParameters& params, DetectionValidator* validator = 0);
	bool DetectLandmarksInVideo(FaceDetectorDlib& face_detector, const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, FaceModelParameters& params, DetectionValidator* validator = 0);
	bool DetectLandmarksInVideo(FaceDetectorDlib& face_detector, const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, const cv::Rect_<double> bounding_box, FaceModelParameters& params, DetectionValidator* validator = 0);

protected:
	CLNF                   mInnerModel;
	CLNF                   mLeftEyeModel;
	CLNF                   mRightEyeModel;
	FaceModelParameters    mInnerParams;
	FaceModelParameters    mLeftEyeParams;
	FaceModelParameters    mRightEyeParams;
};

}
#endif
