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

#ifndef __PDM_h_
#define __PDM_h_

#include <opencv2/core/core.hpp>
#include <FaceModelParameters.h>
#include <FaceGlobalParameters.h>

namespace OpenFace
{

/// <summary>
/// A linear 3D Point Distribution Model (constructed using Non-Rigid structure from motion or PCA)
/// Only describes the model but does not contain an instance of it (no local or global parameters are stored here)
/// Contains the utility functions to help manipulate the model
/// </summary>
class PDM
{
public:    
	/// <summary>
	/// Constructor
	/// </summary>
	PDM() { }

	__forceinline const cv::Mat_<double>& getMeanShape()           const { return mMeanShape;                }
	__forceinline const cv::Mat_<double>& getPrincipleComponents() const { return mPrincipleComponents;      }
	__forceinline const cv::Mat_<double>& getEigenValues()         const { return mEigenValues;              }
	__forceinline const int               getNumberOfPoints()      const { return mMeanShape.rows / 3;       } // Number of vertices
	__forceinline const int               getNumberOfModes()       const { return mPrincipleComponents.cols; } // Listing the number of modes of variation

	/// <summary>
	/// Parser
	/// </summary>
	void Read(std::istream& stream);

	/// <summary>
	///
	/// </summary>
	void Clamp(cv::Mat_<float>& params_local, GlobalParameters& params_global, const FaceModelParameters& params);

	/// <summary>
	/// Compute shape in object space (3D)
	/// </summary>
	void CalcShape3D(cv::Mat_<double>& out_shape, const cv::Mat_<double>& params_local) const;

	/// <summary>
	/// Compute shape in image space (2D)
	/// </summary>
	void CalcShape2D(cv::Mat_<double>& out_shape, const cv::Mat_<double>& params_local, const GlobalParameters& params_global) const;
    
	/// <summary>
	/// Provided the bounding box of a face and the local parameters (with optional rotation),
	/// generates the global parameters that can generate the face with the provided bounding box
	/// </summary>
	void CalcParams(GlobalParameters& out_params_global, const cv::Rect_<double>& bounding_box, const cv::Mat_<double>& params_local, const cv::Vec3d rotation = cv::Vec3d(0.0));

	/// <summary>
	/// Provided the landmark location compute global and local parameters best fitting it 
	/// (can provide optional rotation for potentially better results)
	/// </summary>
	void CalcParams(GlobalParameters& out_params_global, const cv::Mat_<double>& out_params_local, const cv::Mat_<double>& landmark_locations, const cv::Vec3d rotation = cv::Vec3d(0.0));

	/// <summary>
	/// Provided the model parameters, compute the bounding box of a face
	/// </summary>
	cv::Rect CalcBoundingBox(const GlobalParameters& params_global, const cv::Mat_<double>& params_local);

	/// <summary>
	/// Helpers for computing Jacobians, and Jacobians with the weight matrix
	/// </summary>
	void ComputeRigidJacobian(const cv::Mat_<float>& params_local, const GlobalParameters& params_global, cv::Mat_<float> &Jacob, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);
	
	/// <summary>
	/// Helpers for computing Jacobians, and Jacobians with the weight matrix
	/// </summary>
	void ComputeJacobian(const cv::Mat_<float>& params_local, const GlobalParameters& params_global, cv::Mat_<float> &Jacobian, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);

	/// <summary>
	/// Given the current parameters, and the computed delta_p compute the updated parameters
	/// </summary>
	void UpdateModelParameters(const cv::Mat_<float>& delta_p, cv::Mat_<float>& params_local, GlobalParameters& params_global);

protected:
	cv::Mat_<double> mMeanShape;            // The 3D mean shape vector of the PDM [x1,..,xn,y1,...yn,z1,...,zn]
	cv::Mat_<double> mPrincipleComponents;  // Principal components or variation bases of the model, 
	cv::Mat_<double> mEigenValues;          // Eigenvalues (variances) corresponding to the bases
};
}
#endif
