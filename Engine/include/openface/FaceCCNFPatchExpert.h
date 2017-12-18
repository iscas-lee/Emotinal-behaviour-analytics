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

#ifndef __CCNF_PATCH_EXPERT_h_
#define __CCNF_PATCH_EXPERT_h_

#include <opencv2/core/core.hpp>

#include <map>
#include <vector>

namespace OpenFace
{

/// <summary>
/// A single CCNF neuron
/// </summary>
class NeuronCCNF
{
public:
	enum Type
	{
		RAW   = 0,
		GRAD  = 1,
		DEPTH = 3
	};

	/// <summary>
	/// Default constructor
	/// </summary>
	NeuronCCNF() { }

	__forceinline const Type   getNeuronType()  const { return mNeuronType;  }
	__forceinline const double getNormWeights() const { return mNormWeights; }
	__forceinline const double getBias()        const { return mBias;        }
	__forceinline const double getAlpha()       const { return mAlpha;       }

	/// <summary>
	/// Parser
	/// </summary>
	void Read(std::istream& stream);

	// The im_dft, integral_img, and integral_img_sq are precomputed images for convolution speedups (they get set if passed in empty values)
	void Response(cv::Mat_<float> &im, cv::Mat_<double> &im_dft, cv::Mat &integral_img, cv::Mat &integral_img_sq, cv::Mat_<float> &resp);

protected:
	Type            mNeuronType;   // Type of patch (0=raw,1=grad,3=depth, other types besides raw are not actually used now)	
	double          mNormWeights;  // scaling of weights (needed as the energy of neuron might not be 1) 
	double          mBias;         // Weight bias
	double          mAlpha;        // the alpha associated with the neuron
	cv::Mat_<float> mWeights;      // Neural weights

	// can have neural weight dfts that are calculated on the go as needed
	// this allows us not to recompute the dft of the template each time, improving the speed of tracking
	std::map<int, cv::Mat_<double>> mWeightsDfts;
};

//===========================================================================

/// <summary>
/// A CCNF patch expert
/// </summary>
class CCNFPatchExpert
{
public:
	/// <summary>
	/// Default constructor
	/// </summary>
	CCNFPatchExpert() { }

	__forceinline const int    getWidth()           const { return mWidth;           }
	__forceinline const int    getHeight()          const { return mHeight;          }
	__forceinline const double getPatchConfidence() const { return mPatchConfidence; }

	/// <summary>
	/// Parser
	/// </summary>
	void Read(istream& stream, const size_t n_sigmas, const size_t n_betas);

	/// <summary>
	/// Actual work (can pass in an image and a potential depth image, if the CCNF is trained with depth)
	/// </summary>
	void Response(cv::Mat_<float>& area_of_interest, cv::Mat_<float>& response);

	/// <summary>
	/// Compute sigmas for all landmarks for a particular view and window size
	/// </summary>
	void ComputeSigmas(const std::vector<cv::Mat_<float>>& sigma_components, int window_size);

protected:
	int    mWidth;           // Width of the patch expert support region
	int    mHeight;          // Height of the patch expert support region             
	double mPatchConfidence; // How confident we are in the patch

	std::vector<NeuronCCNF>      mNeurons;     // Collection of neurons for this patch expert
	std::vector<int>             mWindowSizes; // Information about the vertex features (association potentials)
	std::vector<cv::Mat_<float>> mSigmas;      // Information about the vertex features (association potentials)
	std::vector<double>          mBetas;       // Information about the vertex features (association potentials)
};
}
#endif
