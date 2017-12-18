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

#ifndef __FACE_VALIDATOR_h_
#define __FACE_VALIDATOR_h_

#include <opencv2/core/core.hpp>
#include <vector>
#include "PAW.h"

using namespace std;

namespace OpenFace
{
/// <summary>
/// Checking if landmark detection was successful using CNN.
/// The regressor outputs -1 for ideal alignment and 1 for worst alignment
/// </summary>
class DetectionValidator
{	
public:    
	
	enum Type
	{
		SVR = 0,
		NN  = 1,
		CNN = 2
	};

	// Default constructor
	DetectionValidator(const string location);

	// Check if the fitting actually succeeded using Convolutional Neural Network
	// Given an image, orientation and detected landmarks, return the result of the appropriate regressor
	double Check(const cv::Vec3d& orientation, const cv::Mat_<uchar>& intensity_img, const cv::Mat_<double>& detected_landmarks);

	// Reading in the model
	//void Read(string location);

protected:
	Type                     mValidatorType;      // What type of validator we're using
	vector<cv::Vec3d>        mOrientations;       // The orientations of each of the landmark detection validator
	vector<PAW>              mPaws;               // Piecewise affine warps to the reference shape (per orientation)
	vector<cv::Mat_<double>> mMeanImages;         // Normalisation for face validation
	vector<cv::Mat_<double>> mStandardDeviations; // Normalisation for face validation

	// CNN layers for each view
	// view -> layer -> input maps -> kernels
	vector<vector<vector<vector<cv::Mat_<float>>>>>             mCNNConvolutionalLayers;
	vector<vector<vector<vector<pair<int, cv::Mat_<double>>>>>> mCNNConvolutionalLayersDft;
	vector<vector<vector<float>>>                               mCNNConvolutionalLayersBias;
	vector<vector<int>>                                         mCNNSubsamplingLayers;
	vector<vector<cv::Mat_<float>>>                             mCNNFullyConnectedLayers;
	vector<vector<float>>                                       mCNNFullyConnectedLayersBias;
	vector<vector<int>>                                         mCNNLayerTypes; // 0 - convolutional, 1 - subsampling, 2 - fully connected

	// Getting the closest view center based on orientation
	size_t GetViewId(const cv::Vec3d& orientation) const;

	// A normalisation helper
	void NormaliseWarpedToVector(const cv::Mat_<double>& warped_img, cv::Mat_<double>& feature_vec, int view_id);
};
}
#endif
