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

// Returns the patch expert responses given a grayscale and an optional depth image.
// Additionally returns the transform from the image coordinates to the response coordinates (and vice versa).
// The computation also requires the current landmark locations to compute response around, the PDM corresponding to the desired model, and the parameters describing its instance
// Also need to provide the size of the area of interest and the desired scale of analysis
void PatchExperts::Response(
	vector<cv::Mat_<float>>& patch_expert_responses, 
	cv::Matx22f&             sim_ref_to_img, 
	cv::Matx22d&             sim_img_to_ref, 
	const cv::Mat_<uchar>&   grayscale_image, 
	const cv::Mat_<float>&   depth_image,
	const PDM&               pdm, 
	const GlobalParameters&  params_global, 
	const cv::Mat_<double>&  params_local, 
	int                      window_size, 
	int                      scale)
{
	const size_t VIEWIDX          = getViewIdx(params_global, scale);		
	const int NUM_PDM_POINTS      = pdm.getNumberOfPoints();
	const int WINDOW_SIZE_SQUARED = window_size * window_size;

	// Initialise the reference shape on which we'll be warping
	GlobalParameters global_ref(mPatchScaling[scale], cv::Vec3d(), 0.0, 0.0);

	// Compute the current landmark locations (around which responses will be computed)
	cv::Mat_<double> landmark_locations;
	cv::Mat_<double> reference_shape;

	// Compute landmark locations and reference shape
	pdm.CalcShape2D(landmark_locations, params_local, params_global);
	pdm.CalcShape2D(reference_shape, params_local, global_ref);
		
	// similarity and inverse similarity transform to and from image and reference shape
	cv::Mat_<double> reference_shape_2D = reference_shape.reshape(1, 2).t();
	cv::Mat_<double> image_shape_2D     = landmark_locations.reshape(1, 2).t();

	sim_img_to_ref = AlignShapesWithScale(image_shape_2D, reference_shape_2D);
	cv::Matx22d sim_ref_to_img_d = sim_img_to_ref.inv(cv::DECOMP_LU);

	const double A1 =  sim_ref_to_img_d(0, 0);
	const double B1 = -sim_ref_to_img_d(0, 1);
		
	sim_ref_to_img(0, 0) = (float)sim_ref_to_img_d(0, 0);
	sim_ref_to_img(0, 1) = (float)sim_ref_to_img_d(0, 1);
	sim_ref_to_img(1, 0) = (float)sim_ref_to_img_d(1, 0);
	sim_ref_to_img(1, 1) = (float)sim_ref_to_img_d(1, 1);

	// Indicates the legal pixels in a depth image, if available (used for CLM-Z area of interest (window) interpolation)
	cv::Mat_<uchar> mask;
	if (!depth_image.empty())
	{
		mask = depth_image > 0;			
		mask = mask / 255;
	}		
	
	// If using CCNF patch experts might need to precalculate Sigmas
	vector<cv::Mat_<float>> sigma_components;

	// Retrieve the correct sigma component size
	for (size_t w_size = 0; w_size < mSigmaComponents.size(); ++w_size)
		if (!mSigmaComponents[w_size].empty())	
			if (WINDOW_SIZE_SQUARED == mSigmaComponents[w_size][0].rows)
				sigma_components = mSigmaComponents[w_size];
			
	// Go through all of the landmarks and compute the Sigma for each
	// Only for visible landmarks
	// Precompute sigmas if they are not computed yet
	for (int lmark = 0; lmark < NUM_PDM_POINTS; lmark++)
		if (mVisibilities[scale][VIEWIDX].at<int>(lmark, 0))
			mCCNFExpertIntensity[scale][VIEWIDX][lmark].ComputeSigmas(sigma_components, window_size);
		
	// calculate the patch responses for every landmark, Actual work happens here.
	tbb::parallel_for(0, NUM_PDM_POINTS, [&](int i) 
	{
		if (mVisibilities[scale][VIEWIDX].rows == NUM_PDM_POINTS &&
			mVisibilities[scale][VIEWIDX].at<int>(i, 0) != 0)
		{
			// Work out how big the area of interest has to be to get a response of window size
			const int WIDTH  = window_size + mCCNFExpertIntensity[scale][VIEWIDX][i].getWidth()  - 1;
			const int HEIGHT = window_size + mCCNFExpertIntensity[scale][VIEWIDX][i].getHeight() - 1;
						
			// scale and rotate to mean shape to reference frame
			cv::Mat sim = (cv::Mat_<float>(2,3) << A1, -B1, landmark_locations.at<double>(i,0), B1, A1, landmark_locations.at<double>(i + NUM_PDM_POINTS, 0));

			// Extract the region of interest around the current landmark location
			cv::Mat_<float> area_of_interest(WIDTH, HEIGHT);

			// Using C style openCV as it does what we need
			CvMat area_of_interest_o = area_of_interest;
			CvMat sim_o = sim;
			IplImage im_o = grayscale_image;			
			cvGetQuadrangleSubPix(&im_o, &area_of_interest_o, &sim_o);
			
			// get the correct size response window			
			patch_expert_responses[i] = cv::Mat_<float>(window_size, window_size);

			// Get intensity response from CCNF patch experts
			mCCNFExpertIntensity[scale][VIEWIDX][i].Response(area_of_interest, patch_expert_responses[i]);
		}	
	});
}

//=============================================================================
// Getting the closest view center based on orientation
const size_t PatchExperts::getViewIdx(const GlobalParameters& params_global, int scale) const
{	
	const size_t NUM_VIEWS = getNumViews(scale);

	size_t idx = 0;
	double dbest;

	for (size_t i = 0; i < NUM_VIEWS; i++)
	{
		const double v1 = params_global.orient[0] - mCenters[scale][i][0]; 
		const double v2 = params_global.orient[1] - mCenters[scale][i][1];
		const double v3 = params_global.orient[2] - mCenters[scale][i][2];
			
		const double d = v1*v1 + v2*v2 + v3*v3;

		if (i == 0 || d < dbest)
		{
			dbest = d;
			idx = i;
		}
	}

	return idx;
}

//===========================================================================
void PatchExperts::Read(const vector<string>& ccnf_expert_locations)
{
	// Initialise and read CCNF patch experts 
	const size_t NUM_CCNF = ccnf_expert_locations.size();
	
	mCenters.resize(NUM_CCNF);
	mVisibilities.resize(NUM_CCNF);
	mPatchScaling.resize(NUM_CCNF);
	mCCNFExpertIntensity.resize(NUM_CCNF);

	for (size_t scale = 0; scale < NUM_CCNF; ++scale)
	{	
		// file
		const string& location = ccnf_expert_locations[scale];

		cout << "Reading the intensity CCNF patch experts from: " << location << "....";

		// parse the binary ccnf patch expert file
		ifstream stream(location, ios::in | ios::binary);
		if (stream.is_open())
		{
			ReadCCNFPatchExpert(
				stream,
				mCenters[scale],
				mVisibilities[scale],
				mCCNFExpertIntensity[scale],
				mPatchScaling[scale]);

			stream.close();
			cout << "Done" << endl;
		}
		else		
			cout << "Can't find/open the patches file" << endl;		
	}
}

//======================= Reading the CCNF patch experts =========================================//
void PatchExperts::ReadCCNFPatchExpert(
	std::istream&                              stream, 
	std::vector<cv::Vec3d>&                    centers, 
	std::vector<cv::Mat_<int>>&                visibility, 
	std::vector<std::vector<CCNFPatchExpert>>& patches, 
	double&                                    patchScaling)
{
	int numberViews;
		
	stream.read ((char*)&patchScaling, 8);
	stream.read ((char*)&numberViews, 4);

	// read the visibility
	centers.resize(numberViews);
	visibility.resize(numberViews);  
	patches.resize(numberViews);
		
	// centers of each view (which view corresponds to which orientation)
	for (size_t i = 0; i < centers.size(); i++)
	{
		cv::Mat center;
		ReadMatBin(stream, center);
		center.copyTo(centers[i]);
		centers[i] = centers[i] * M_PI / 180.0;
	}

	// the visibility of points for each of the views (which verts are visible at a specific view
	for (size_t i = 0; i < visibility.size(); i++)		
		ReadMatBin(stream, visibility[i]);
		
	const int NUM_POINTS = visibility[0].rows;

	// Read the possible SigmaInvs (without beta), 
	// this will be followed by patch reading 
	// (this assumes all of them have the same type, and number of betas)
	int num_win_sizes;
	stream.read ((char*)&num_win_sizes, 4);

	vector<int> windows;

	windows.resize(num_win_sizes);
	mSigmaComponents.resize(num_win_sizes);

	for (int w = 0; w < num_win_sizes; ++w)
	{
		int num_sigma_comp;

		stream.read ((char*)&windows[w], 4);
		stream.read ((char*)&num_sigma_comp, 4);

		mSigmaComponents[w].resize(num_sigma_comp);

		for (int s = 0; s < num_sigma_comp; ++s)		
			ReadMatBin(stream, mSigmaComponents[w][s]);
	}

	// read the patches themselves
	for (size_t i = 0; i < patches.size(); i++)
	{
		// number of patches for each view
		patches[i].resize((size_t)NUM_POINTS);

		// read in each patch
		for (int j = 0; j < NUM_POINTS; j++)
			patches[i][j].Read(stream, windows.size(), mSigmaComponents[0].size());
	}
}
