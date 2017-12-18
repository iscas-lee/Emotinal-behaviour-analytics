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

//===========================================================================
void NeuronCCNF::Read(istream& stream)
{
	// Sanity check
	int read_type;
	stream.read ((char*)&read_type, 4);
	assert(read_type == 2);

	// Read and check Type
	int type;
	stream.read ((char*)&type, 4);
	mNeuronType = (Type)type;
	assert(mNeuronType == RAW);

	stream.read ((char*)&mNormWeights, 8);
	stream.read ((char*)&mBias, 8);
	stream.read ((char*)&mAlpha, 8);
	
	ReadMatBin(stream, mWeights); 
}

//===========================================================================
void NeuronCCNF::Response(cv::Mat_<float> &im, cv::Mat_<double> &im_dft, cv::Mat &integral_img, cv::Mat &integral_img_sq, cv::Mat_<float> &resp)
{
	int h = im.rows - mWeights.rows + 1;
	int w = im.cols - mWeights.cols + 1;
	
	if(resp.empty())				
		resp.create(h, w);
	
	// the linear multiplication, efficient calc of response
	matchTemplate_m(im, im_dft, integral_img, integral_img_sq, mWeights, mWeightsDfts, resp, CV_TM_CCOEFF_NORMED); 
	
	cv::MatIterator_<float> p  = resp.begin();
	cv::MatIterator_<float> q1 = resp.begin(); // respone for each pixel
	cv::MatIterator_<float> q2 = resp.end();

	// precomputed before loop
	const double twoAlpha = 2.0 * mAlpha;

	// the logistic function (sigmoid) applied to the response
	while(q1 != q2)
	{
		*p++ = twoAlpha / (1.0 + exp( -(*q1++ * mNormWeights + mBias )));
	}
}

//===========================================================================
//===========================================================================
//===========================================================================
void CCNFPatchExpert::Read(istream& stream, const size_t n_sigmas, const size_t n_betas)
{
	// Sanity check
	int read_type;

	stream.read ((char*)&read_type, 4);
	assert(read_type == 5);

	// the number of neurons for this patch
	int num_neurons;
	stream.read ((char*)&mWidth, 4);
	stream.read ((char*)&mHeight, 4);
	stream.read ((char*)&num_neurons, 4);

	if (num_neurons == 0)
	{
		// empty patch due to landmark being invisible at that orientation	
		// read an empty int (due to the way things were written out)
		stream.read ((char*)&num_neurons, 4);
		return;
	}

	// read neurons
	mNeurons.resize(num_neurons);
	for (int i = 0; i < num_neurons; i++)
		mNeurons[i].Read(stream);

	if (n_sigmas > 0)
	{
		mBetas.resize(n_betas);

		// read beta-values
		for (size_t i = 0; i < n_betas; ++i)		
			stream.read ((char*)&mBetas[i], 8);		
	}	

	// Read the patch confidence
	stream.read ((char*)&mPatchConfidence, 8);
}

//===========================================================================
void CCNFPatchExpert::Response(cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
{
	int response_height = area_of_interest.rows - mHeight + 1;
	int response_width  = area_of_interest.cols - mWidth  + 1;

	assert(response.rows == response_height);
	assert(response.cols == response_width);

	// init response with zeros
	response.setTo(0);
	
	//////////////////////////////////////////////////////////////

	// the placeholder for the DFT of the image, the integral image, 
	// and squared integral image so they don't get recalculated for every response
	cv::Mat_<double> area_of_interest_dft;
	cv::Mat integral_image;
	cv::Mat integral_image_sq;
	cv::Mat_<float> neuron_response;

	// responses from the neural layers
	for(size_t i = 0; i < mNeurons.size(); i++)
	{		
		// Do not bother with neuron response if the alpha is tiny and 
		// will not contribute much to overall result
		if(mNeurons[i].getAlpha() > 1e-4)
		{
			// calculate the response of the neuron
			mNeurons[i].Response(
				area_of_interest, 
				area_of_interest_dft, 
				integral_image, 
				integral_image_sq, 
				neuron_response);

			response = response + neuron_response;						
		}
	}

	//////////////////////////////////////////////////////////////

	// Find the matching sigma
	int sigmaIdx = -1;
	for(size_t i=0; i < mWindowSizes.size(); ++i)
	{
		if(mWindowSizes[i] == response_height)
		{
			// Found the correct sigma
			sigmaIdx = (int)i;
			break;
		}
	}

	assert(sigmaIdx >= 0 && sigmaIdx < mSigmas.size());

	//////////////////////////////////////////////////////////////

	cv::Mat_<float> resp_vec_f = response.reshape(1, response_height * response_width);
	cv::Mat out                = mSigmas[sigmaIdx] * resp_vec_f;
	
	response = out.reshape(1, response_height);

	// Making sure the response does not have negative numbers
	double min;
	minMaxIdx(response, &min, 0);

	if(min < 0.0)
		response = response - min;	
}

//===========================================================================
void CCNFPatchExpert::ComputeSigmas(const std::vector<cv::Mat_<float>>& sigma_components, int window_size)
{
	for (size_t i = 0; i < mWindowSizes.size(); ++i)	
		if (mWindowSizes[i] == window_size)
			return;

	// Each of the landmarks will have the same connections, hence constant number of sigma components
	const size_t n_betas  = sigma_components.size();
	const size_t n_alphas = mNeurons.size();

	const int window_size_squared = window_size * window_size;

	// calculate the sigmas based on alphas and betas
	float sum_alphas = 0;

	// sum the alphas first
	for (size_t a = 0; a < n_alphas; ++a)
		sum_alphas += (float)mNeurons[a].getAlpha();	

	cv::Mat_<float> q1 = sum_alphas * cv::Mat_<float>::eye(window_size_squared, window_size_squared);
	cv::Mat_<float> q2 = cv::Mat_<float>::zeros(window_size_squared, window_size_squared);
	
	for (size_t b = 0; b < n_betas; ++b)	
		q2 = q2 + ((float)this->mBetas[b]) * sigma_components[b];
	
	cv::Mat_<float> sigmaInv = 2.0f * (q1 + q2);
	cv::Mat sigma_f;

	cv::invert(sigmaInv, sigma_f, cv::DECOMP_CHOLESKY);

	// store it
	mWindowSizes.push_back(window_size);
	mSigmas.push_back(sigma_f);
}
