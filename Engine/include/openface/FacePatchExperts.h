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

#ifndef __FACE_PATCHEXPERTS_h_
#define __FACE_PATCHEXPERTS_h_

#include <opencv2/core/core.hpp>
#include "FaceGlobalParameters.h"
#include "FaceCCNFPatchExpert.h"
#include "PDM.h"

namespace OpenFace
{
/// <summary>
/// Manages the multi-dimensional layed out CCNFPatchExperts.
/// </summary>
class PatchExperts
{
public:
	/// <summary>
	/// Default Constructor
	/// </summary>
	PatchExperts() { }

	__forceinline const vector<vector<vector<CCNFPatchExpert>>>& getCCNFExpertIntensity()         const { return mCCNFExpertIntensity;   }
	__forceinline const vector<double>&                          getPatchScaling()                const { return mPatchScaling;          }
	__forceinline const vector<vector<cv::Mat_<int>>>&           getVisibilities()                const { return mVisibilities;          }
	__forceinline const size_t                                   getNumViews(const int scale = 0) const { return mCenters[scale].size(); }

	// Returns the patch expert responses given a grayscale and an optional depth image.
	// Additionally returns the transform from the image coordinates to the response coordinates (and vice versa).
	// The computation also requires the current landmark locations to compute response around, the PDM corresponding to the desired model, and the parameters describing its instance
	// Also need to provide the size of the area of interest and the desired scale of analysis
	void Response(
		vector<cv::Mat_<float>>& patch_expert_responses, 
		cv::Matx22f&             sim_ref_to_img, 
		cv::Matx22d&             sim_img_to_ref, 
		const cv::Mat_<uchar>&   grayscale_image, 
		const cv::Mat_<float>&   depth_image,
		const PDM&               pdm, 
		const GlobalParameters&  params_global, 
		const cv::Mat_<double>&  params_local, 
		int                      window_size,
		int                      scale);

	// Getting the best view associated with the current orientation
	const size_t getViewIdx(const GlobalParameters& params_global, int scale) const;

	// The number of views at a particular scale
	
	// Reading in all of the patch experts
	void Read(const vector<string>& ccnf_expert_locations);

protected:
	vector<vector<vector<CCNFPatchExpert>>> mCCNFExpertIntensity; // The collection of LNF (CCNF) patch experts (for intensity images), the experts are laid out scale->view->landmark	
	vector<vector<cv::Mat_<float>>>         mSigmaComponents;     // The node connectivity for CCNF experts, at different window sizes and corresponding to separate edge features
	vector<double>                          mPatchScaling;        // The available scales for intensity patch experts
	vector<vector<cv::Vec3d>>               mCenters;             // The available views for the patch experts at every scale (in radians)	
	vector<vector<cv::Mat_<int>>>           mVisibilities;        // Landmark visibilities for each scale and view

	void ReadCCNFPatchExpert(
		std::istream&                              stream,
		std::vector<cv::Vec3d>&                    centers, 
		std::vector<cv::Mat_<int>>&                visibility, 
		std::vector<std::vector<CCNFPatchExpert>>& patches, 
		double&                                    patchScaling);
};
}
#endif
