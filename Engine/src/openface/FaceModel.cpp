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

const vector<pair<int, int>> LEFT_EYE_MAPPING =
{
	pair<int, int>(36, 8),
	pair<int, int>(37, 10),
	pair<int, int>(38, 12),
	pair<int, int>(39, 14),
	pair<int, int>(40, 16),
	pair<int, int>(41, 18)
};

const vector<pair<int, int>> RIGHT_EYE_MAPPING =
{
	pair<int, int>(42, 8),
	pair<int, int>(43, 10),
	pair<int, int>(44, 12),
	pair<int, int>(45, 14),
	pair<int, int>(46, 16),
	pair<int, int>(47, 18)
};

const vector<pair<int, int>> INNER_MAPPING =
{
	pair<int, int>(17, 0),  pair<int, int>(18, 1),
	pair<int, int>(19, 2),  pair<int, int>(20, 3),
	pair<int, int>(21, 4),  pair<int, int>(22, 5),
	pair<int, int>(23, 6),  pair<int, int>(24, 7),
	pair<int, int>(25, 8),  pair<int, int>(26, 9),
	pair<int, int>(27, 10), pair<int, int>(28, 11),
	pair<int, int>(29, 12), pair<int, int>(30, 13),
	pair<int, int>(31, 14), pair<int, int>(32, 15),
	pair<int, int>(33, 16), pair<int, int>(34, 17),
	pair<int, int>(35, 18), pair<int, int>(36, 19),
	pair<int, int>(37, 20), pair<int, int>(38, 21),
	pair<int, int>(39, 22), pair<int, int>(40, 23),
	pair<int, int>(41, 24), pair<int, int>(42, 25),
	pair<int, int>(43, 26), pair<int, int>(44, 27),
	pair<int, int>(45, 28), pair<int, int>(46, 29),
	pair<int, int>(47, 30), pair<int, int>(48, 31),
	pair<int, int>(49, 32), pair<int, int>(50, 33),
	pair<int, int>(51, 34), pair<int, int>(52, 35),
	pair<int, int>(53, 36), pair<int, int>(54, 37),
	pair<int, int>(55, 38), pair<int, int>(56, 39),
	pair<int, int>(57, 40), pair<int, int>(58, 41),
	pair<int, int>(59, 42), pair<int, int>(60, 43),
	pair<int, int>(61, 44), pair<int, int>(62, 45),
	pair<int, int>(63, 46), pair<int, int>(64, 47),
	pair<int, int>(65, 48), pair<int, int>(66, 49),
	pair<int, int>(67, 50)
};

//=============================================================================
//=============================================================================

CLNF::CLNF(
	const string&         mainPdmFile,
	const vector<string>& mainCcnfFiles,
	const string&         mainTriangulationsFile,
	const Type            modelType) : 
	mType(modelType),
	mDetectionSuccess(false),
	mTrackingInitialised(false),
	mModelLikelihood(-10.0),  // very low
	mDetectionCertainty(1.0), // very uncertain
	mFailuresInARow(-1)
{
	// read only main model
	Read(mainPdmFile, mainCcnfFiles, mainTriangulationsFile);

	/////////////////////////////////////////////////////

	// init landmarks
	mDetectedLandmarks.create(2 * mPDM.getNumberOfPoints(), 1);
	mDetectedLandmarks.setTo(0);

	// local parameters (shape)
	mParamsLocal.create(mPDM.getNumberOfModes(), 1);
	mParamsLocal.setTo(0.0);

	// reset global parameters
	mParamsGlobal.reset();
}

void CLNF::Read(const string& pdmFile, const vector<string>& ccnfFiles, const string& triangulationsFile)
{
	///////////////////////////////////////////////////////////////////////////////////////
	// Read the PDM data
	cout << "Reading the PDM module from: " << pdmFile << "....";
	ifstream pdmstream(pdmFile, ios_base::in);
	mPDM.Read(pdmstream);
	pdmstream.close();
	cout << "Done" << endl;

	///////////////////////////////////////////////////////////////////////////////////////
	// Read Triangulations
	if (triangulationsFile.compare("") != 0)
	{
		cout << "Reading the Triangulations module from: " << triangulationsFile << "....";
		ifstream stream(triangulationsFile, ios_base::in);

		SkipComments(stream);

		int numViews;
		stream >> numViews;

		// read in the triangulations
		mTriangulations.resize(numViews);

		for (int i = 0; i < numViews; ++i)
		{
			SkipComments(stream);
			ReadMat(stream, mTriangulations[i]);
		}
		cout << "Done" << endl;
		stream.close();
	}

	///////////////////////////////////////////////////////////////////////////////////////
	// Read CCNF
	mPatchExperts.Read(ccnfFiles);
}

// Resetting the model (for a new video, or complet reinitialisation
void CLNF::Reset()
{
	mDetectedLandmarks.setTo(0);

	mDetectionSuccess    = false;
	mTrackingInitialised = false;
	mModelLikelihood     = -10;  // very low
	mDetectionCertainty  =  1;   // very uncertain
	mFailuresInARow      = -1;

	// local parameters (shape)
	mParamsLocal.setTo(0.0);

	// global parameters (pose) [scale, euler_x, euler_y, euler_z, tx, ty]
	mParamsGlobal.reset();

	mFaceTemplate = cv::Mat_<uchar>();
}

// The main internal landmark detection call (should not be used externally?)
bool CLNF::DetectLandmarks(const cv::Mat_<uchar>& image, const cv::Mat_<float>& depth, FaceModelParameters& params)
{
	// Fits from the current estimate of local and global parameters in the model
	const bool FIT_SUCCESS = Fit(image, depth, params.window_sizes_current, params);

	// Store the landmarks converged on in detected_landmarks
	calcShape2D();

	mDetectionSuccess   = FIT_SUCCESS;
	mDetectionCertainty = FIT_SUCCESS ? -1.0 : 1.0;
	
	return mDetectionSuccess;
}

//=============================================================================
bool CLNF::Fit(const cv::Mat_<uchar>& im, const cv::Mat_<float>& depthImg, const std::vector<int>& window_sizes, const FaceModelParameters& parameters)
{
	// Making sure it is a single channel image
	assert(im.channels() == 1);	
	
	// Placeholder for the landmarks
	cv::Mat_<double> current_shape(2 * mPDM.getNumberOfPoints() , 1, 0.0);

	int n = mPDM.getNumberOfPoints();
	
	cv::Mat_<float> depth_img_no_background;
	
	// Background elimination from the depth image
	if (!depthImg.empty())
	{
		// The attempted background removal can fail leading to tracking failure
		if (!RemoveBackground(depth_img_no_background, depthImg))
			return false;	
	}

	int num_scales = mPatchExperts.getPatchScaling().size();

	// Storing the patch expert response maps
	vector<cv::Mat_<float> > patch_expert_responses(n);

	// Converting from image space to patch expert space (normalised for rotation and scale)
	cv::Matx22f sim_ref_to_img;
	cv::Matx22d sim_img_to_ref;

	FaceModelParameters tmp_parameters = parameters;

	// Optimise the model across a number of areas of interest (usually in descending window size and ascending scale size)
	for (int scale = 0; scale < num_scales; scale++)
	{
		int window_size = window_sizes[scale];

		// get reference to patch expert scaling
		const vector<double>& PATCHSCALING = mPatchExperts.getPatchScaling();

		if (window_size == 0 ||  0.9 * PATCHSCALING[scale] > mParamsGlobal.scale)
			continue;

		// The patch expert response computation
		if (scale != window_sizes.size() - 1)
		{
			mPatchExperts.Response(
				patch_expert_responses, 
				sim_ref_to_img, 
				sim_img_to_ref, 
				im, 
				depth_img_no_background, 
				mPDM, 
				mParamsGlobal, 
				mParamsLocal, 
				window_size, 
				scale);
		}
		else
		{
			// Do not use depth for the final iteration as it is not as accurate
			mPatchExperts.Response(
				patch_expert_responses, 
				sim_ref_to_img, 
				sim_img_to_ref, 
				im, 
				cv::Mat(), 
				mPDM, 
				mParamsGlobal, 
				mParamsLocal, 
				window_size, 
				scale);
		}
		
		if (parameters.refine_parameters == true)
		{
			// Adapt the parameters based on scale (wan't to reduce regularisation as scale increases, but increa sigma and tikhonov)
			tmp_parameters.reg_factor = parameters.reg_factor - 15 * log(PATCHSCALING[scale]/0.25)/log(2);
			
			if(tmp_parameters.reg_factor <= 0)
				tmp_parameters.reg_factor = 0.001;

			tmp_parameters.sigma = parameters.sigma + 0.25 * log(PATCHSCALING[scale]/0.25)/log(2);
			tmp_parameters.weight_factor = parameters.weight_factor + 2 * parameters.weight_factor *  log(PATCHSCALING[scale]/0.25)/log(2);
		}

		// Get the current landmark locations
		mPDM.CalcShape2D(current_shape, mParamsLocal, mParamsGlobal);

		// Get the view used by patch experts
		size_t view_id = mPatchExperts.getViewIdx(mParamsGlobal, scale);

		// the actual optimisation step
		this->NU_RLMS(
			mParamsGlobal,
			mParamsLocal,
			patch_expert_responses, 
			GlobalParameters(mParamsGlobal),
			mParamsLocal.clone(),
			current_shape, 
			sim_img_to_ref, 
			sim_ref_to_img, 
			window_size, 
			view_id, 
			true, 
			scale, 
			mLandmarkLikelihoods,
			tmp_parameters);

		// non-rigid optimisation
		mModelLikelihood = NU_RLMS(
			mParamsGlobal,
			mParamsLocal,
			patch_expert_responses, 
			GlobalParameters(mParamsGlobal),
			mParamsLocal.clone(),
			current_shape, 
			sim_img_to_ref, 
			sim_ref_to_img, 
			window_size, 
			view_id, 
			false, 
			scale, 
			mLandmarkLikelihoods,
			tmp_parameters);
		
		// Can't track very small images reliably (less than ~30px across)
		if(mParamsGlobal.scale < 0.25)
		{
			cout << "Face too small for landmark detection" << endl;
			return false;
		}
	}

	return true;
}

//=============================================================================
void CLNF::NonVectorisedMeanShift_precalc_kde(
	cv::Mat_<float>&               out_mean_shifts, 
	const vector<cv::Mat_<float>>& patch_expert_responses, 
	const cv::Mat_<float>&         dxs, 
	const cv::Mat_<float>&         dys, 
	int                            resp_size, 
	float                          a, 
	int                            scale, 
	int                            view_id, 
	map<int, cv::Mat_<float>>&     kde_resp_precalc)
{	
	int n = dxs.rows;
	
	cv::Mat_<float> kde_resp;
	float step_size = 0.1;

	// if this has not been precomputer, precompute it, otherwise use it
	if (kde_resp_precalc.find(resp_size) == kde_resp_precalc.end())
	{		
		kde_resp = cv::Mat_<float>((int)((resp_size / step_size)*(resp_size/step_size)), resp_size * resp_size);
		cv::MatIterator_<float> kde_it = kde_resp.begin();

		for(int x = 0; x < resp_size/step_size; x++)
		{
			float dx = x * step_size;
			for(int y = 0; y < resp_size/step_size; y++)
			{
				float dy = y * step_size;

				int ii,jj;
				float v,vx,vy;
			
				for(ii = 0; ii < resp_size; ii++)
				{
					vx = (dy-ii)*(dy-ii);
					for(jj = 0; jj < resp_size; jj++)
					{
						vy = (dx-jj)*(dx-jj);

						// the KDE evaluation of that point
						v = exp(a*(vx+vy));
						
						*kde_it++ = v;
					}
				}
			}
		}

		kde_resp_precalc[resp_size] = kde_resp.clone();
	}
	else
	{
		// use the precomputed version
		kde_resp = kde_resp_precalc.find(resp_size)->second;
	}

	// get reference to visibilities
	const vector<vector<cv::Mat_<int>>>& VISIBILITIES = 
		mPatchExperts.getVisibilities();

	// for every point (patch) calculating mean-shift
	for (int i = 0; i < n; i++)
	{
		if (VISIBILITIES[scale][view_id].at<int>(i,0) == 0)
		{
			out_mean_shifts.at<float>(i,0) = 0;
			out_mean_shifts.at<float>(i+n,0) = 0;
			continue;
		}

		// indices of dx, dy
		float dx = dxs.at<float>(i);
		float dy = dys.at<float>(i);

		// Ensure that we are within bounds (important for precalculation)
		if(dx < 0)
			dx = 0;
		if(dy < 0)
			dy = 0;
		if(dx > resp_size - step_size)
			dx = resp_size - step_size;
		if(dy > resp_size - step_size)
			dy = resp_size - step_size;
		
		// Pick the row from precalculated kde that approximates the current dx, dy best		
		int closest_col = (int)(dy /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast
		int closest_row = (int)(dx /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast
		
		int idx = closest_row * ((int)(resp_size/step_size + 0.5)) + closest_col; // Plus 0.5 is there, as C++ rounds down with int cast

		cv::MatIterator_<float> kde_it = kde_resp.begin() + kde_resp.cols*idx;
		
		float mx=0.0;
		float my=0.0;
		float sum=0.0;

		// Iterate over the patch responses here
		cv::MatConstIterator_<float> p = patch_expert_responses[i].begin();
			
		for(int ii = 0; ii < resp_size; ii++)
		{
			for(int jj = 0; jj < resp_size; jj++)
			{

				// the KDE evaluation of that point multiplied by the probability at the current, xi, yi
				float v = (*p++) * (*kde_it++);

				sum += v;

				// mean shift in x and y
				mx += v*jj;
				my += v*ii; 

			}
		}
		
		float msx = (mx/sum - dx);
		float msy = (my/sum - dy);

		out_mean_shifts.at<float>(i,0) = msx;
		out_mean_shifts.at<float>(i+n,0) = msy;
	}
}

void CLNF::GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const FaceModelParameters& parameters)
{
	const int N = mPDM.getNumberOfPoints();

	// Is the weight matrix needed at all
	if (parameters.weight_factor > 0)
	{
		WeightMatrix = cv::Mat_<float>::zeros(N*2, N*2);

		// reference to the experts for given scale and view
		const vector<CCNFPatchExpert>& EXPERTS =
			mPatchExperts.getCCNFExpertIntensity()[scale][view_id];

		// iterate experts
		for (int p = 0; p < N; p++)
		{		
			// for the x dimension
			WeightMatrix.at<float>(p,p) = WeightMatrix.at<float>(p,p)  + EXPERTS[p].getPatchConfidence();
				
			// for they y dimension
			WeightMatrix.at<float>(p+N,p+N) = WeightMatrix.at<float>(p,p);		
		}

		WeightMatrix = parameters.weight_factor * WeightMatrix;
	}
	else
	{
		WeightMatrix = cv::Mat_<float>::eye(N*2, N*2);
	}
}

//=============================================================================
double CLNF::NU_RLMS(
	GlobalParameters&              final_global, 
	cv::Mat_<double>&              final_local, 
	const vector<cv::Mat_<float>>& patch_expert_responses, 
	const GlobalParameters&        initial_global, 
	const cv::Mat_<double>&        initial_local,
	const cv::Mat_<double>&        base_shape, 
	const cv::Matx22d&             sim_img_to_ref, 
	const cv::Matx22f&             sim_ref_to_img, 
	int                            resp_size, 
	int                            view_id, 
	bool                           rigid, 
	int                            scale, 
	cv::Mat_<double>&              landmark_lhoods,
	const FaceModelParameters&     parameters)
{
	// get some stuff from our PDM
	const int NUM_POINTS      = mPDM.getNumberOfPoints();
	const int m               = mPDM.getNumberOfModes();
	const cv::Mat_<double>& M = mPDM.getMeanShape();
	const cv::Mat_<double>& E = mPDM.getEigenValues();
	
	/////////////////////////////////////////////////////////////

	GlobalParameters current_global(initial_global);

	cv::Mat_<float> current_local;
	initial_local.convertTo(current_local, CV_32F);

	// Pre-calculate the regularisation term
	cv::Mat_<float> regTerm;

	if (rigid)
		regTerm = cv::Mat_<float>::zeros(6, 6);
	
	else
	{
		cv::Mat_<double> regularisations = cv::Mat_<double>::zeros(1, 6 + m);

		// Setting the regularisation to the inverse of eigenvalues
		cv::Mat(parameters.reg_factor / E).copyTo(regularisations(cv::Rect(6, 0, m, 1)));
		cv::Mat_<double> regTerm_d = cv::Mat::diag(regularisations.t());
		regTerm_d.convertTo(regTerm, CV_32F);
	}	

	cv::Mat_<float> WeightMatrix;
	GetWeightMatrix(WeightMatrix, scale, view_id, parameters);

	cv::Mat_<float> dxs, dys;
	
	// The preallocated memory for the mean shifts
	cv::Mat_<float> mean_shifts(2 * mPDM.getNumberOfPoints(), 1, 0.0);

	// get reference to visibilities
	const vector<vector<cv::Mat_<int>>>& VISIBILITIES =
		mPatchExperts.getVisibilities();
	
	/////////////////////////////////////////////////////////////////////////////

	cv::Mat_<double> current_shape;
	cv::Mat_<double> previous_shape;

	// Number of iterations
	for (int iter = 0; iter < parameters.num_optimisation_iteration; iter++)
	{
		// get the current estimates of x
		mPDM.CalcShape2D(current_shape, current_local, current_global);
		
		// if the shape hasn't changed terminate
		if (iter > 0 && cv::norm(current_shape, previous_shape) < 0.01)		
			break;

		current_shape.copyTo(previous_shape);
		
		////////////////////////////////////////////////////////

		// Jacobian, and transposed weighted jacobian
		cv::Mat_<float> J, J_w_t;

		// calculate the appropriate Jacobians in 2D, 
		// even though the actual behaviour is in 3D, 
		// using small angle approximation and oriented shape
		if (rigid)
			mPDM.ComputeRigidJacobian(current_local, current_global, J, WeightMatrix, J_w_t);

		else
			mPDM.ComputeJacobian(current_local, current_global, J, WeightMatrix, J_w_t);
		
		////////////////////////////////////////////////////////

		// useful for mean shift calculation
		const float A = -0.5f / (float)(parameters.sigma * parameters.sigma);

		cv::Mat_<double> current_shape_2D = current_shape.reshape(1, 2).t();
		cv::Mat_<double> base_shape_2D = base_shape.reshape(1, 2).t();

		cv::Mat_<float> offsets;
		cv::Mat((current_shape_2D - base_shape_2D) * cv::Mat(sim_img_to_ref).t()).convertTo(offsets, CV_32F);
		
		dxs = offsets.col(0) + (resp_size-1)/2;
		dys = offsets.col(1) + (resp_size-1)/2;
		
		// call NonVectorisedMeanShift_precalc_kde
		NonVectorisedMeanShift_precalc_kde(
			mean_shifts, patch_expert_responses, dxs, dys, resp_size, A, scale, view_id, mKdeRespPrecalc);

		// Now transform the mean shifts to the the image reference frame, 
		// as opposed to one of ref shape (object space)
		cv::Mat_<float> mean_shifts_2D = (mean_shifts.reshape(1, 2)).t();
		
		mean_shifts_2D = mean_shifts_2D * cv::Mat(sim_ref_to_img).t();
		mean_shifts = cv::Mat(mean_shifts_2D.t()).reshape(1, NUM_POINTS * 2);

		// remove non-visible observations
		for (int i = 0; i < NUM_POINTS; ++i)
		{
			// if patch unavailable for current index
			if (VISIBILITIES[scale][view_id].at<int>(i,0) == 0)
			{				
				cv::Mat Jx = J.row(i);
				Jx = cvScalar(0);
				
				cv::Mat Jy = J.row(i + NUM_POINTS);
				Jy = cvScalar(0);
				
				mean_shifts.at<float>(i, 0) = 0.0f;
				mean_shifts.at<float>(i + NUM_POINTS, 0) = 0.0f;
			}
		}

		// projection of the meanshifts onto the jacobians (using the weighted Jacobian, see Baltrusaitis 2013)
		cv::Mat_<float> J_w_t_m = J_w_t * mean_shifts;

		// Add the regularisation term
		if (!rigid)	
			J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) - regTerm(cv::Rect(6,6, m, m)) * current_local;		

		// Calculating the Hessian approximation
		cv::Mat_<float> Hessian = J_w_t * J;

		// Add the Tikhonov regularisation
		Hessian = Hessian + regTerm;

		// Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
		cv::Mat_<float> param_update;
		cv::solve(Hessian, J_w_t_m, param_update, CV_CHOLESKY);
		
		// update the reference
		mPDM.UpdateModelParameters(param_update, current_local, current_global);
		
		// clamp to the local parameters for valid expressions
		mPDM.Clamp(current_local, current_global, parameters);
	}

	/////////////////////////////////////////////////////////////////////////////

	// compute the log likelihood
	double loglhood = 0;
	
	landmark_lhoods = cv::Mat_<double>(NUM_POINTS, 1, -1e8);
	
	for (int i = 0; i < NUM_POINTS; i++)
	{
		if (VISIBILITIES[scale][view_id].at<int>(i,0) == 0 )		
			continue;
		
		float dx = dxs.at<float>(i);
		float dy = dys.at<float>(i);

		int ii,jj;
		float v,vx,vy,sum=0.0;

		// Iterate over the patch responses here
		cv::MatConstIterator_<float> p = patch_expert_responses[i].begin();
			
		for (ii = 0; ii < resp_size; ii++)
		{
			vx = (dy-ii)*(dy-ii);
			for (jj = 0; jj < resp_size; jj++)
			{
				vy = (dx-jj)*(dx-jj);

				// the probability at the current, xi, yi
				v = *p++;

				// the KDE evaluation of that point
				v *= exp(-0.5*(vx+vy)/(parameters.sigma * parameters.sigma));

				sum += v;
			}
		}

		landmark_lhoods.at<double>(i,0) = (double)sum;

		// the offset is there for numerical stability
		loglhood += log(sum + 1e-8);
	}

	loglhood = loglhood/sum(VISIBILITIES[scale][view_id])[0];

	final_global = current_global;
	final_local = current_local;

	return loglhood;
}

bool CLNF::RemoveBackground(cv::Mat_<float>& out_depth_image, const cv::Mat_<float>& depth_image)
{
	// use the current estimate of the face location to determine what is foreground and background
	const double tx = mParamsGlobal.tx;
	const double ty = mParamsGlobal.ty;

	// if we are too close to the edge fail
	if(tx - 9 <= 0 || ty - 9 <= 0 || tx + 9 >= depth_image.cols || ty + 9 >= depth_image.rows)
	{
		cout << "Face estimate is too close to the edge, tracking failed" << endl;
		return false;
	}

	cv::Mat_<double> current_shape;

	mPDM.CalcShape2D(current_shape, mParamsLocal, mParamsGlobal);

	double min_x, max_x, min_y, max_y;

	int n = this->mPDM.getNumberOfPoints();

	cv::minMaxLoc(current_shape(cv::Range(0, n), cv::Range(0,1)), &min_x, &max_x);
	cv::minMaxLoc(current_shape(cv::Range(n, n*2), cv::Range(0,1)), &min_y, &max_y);

	// the area of interest: size of face with some scaling ( these scalings are fairly ad-hoc)
	double width = 3 * (max_x - min_x); 
	double height = 2.5 * (max_y - min_y); 

	// getting the region of interest from the depth image,
	// so we don't get other objects lying at same depth as head in the image but away from it
	cv::Rect_<int> roi((int)(tx-width/2), (int)(ty - height/2), (int)width, (int)height);

	// clamp it if it does not lie fully in the image
	if(roi.x < 0) roi.x = 0;
	if(roi.y < 0) roi.y = 0;
	if(roi.width + roi.x >= depth_image.cols) roi.x = depth_image.cols - roi.width;
	if(roi.height + roi.y >= depth_image.rows) roi.y = depth_image.rows - roi.height;
		
	if(width > depth_image.cols)
	{
		roi.x = 0; roi.width = depth_image.cols;
	}
	if(height > depth_image.rows)
	{
		roi.y = 0; roi.height = depth_image.rows;
	}

	if(roi.width == 0) roi.width = depth_image.cols;
	if(roi.height == 0) roi.height = depth_image.rows;

	if(roi.x >= depth_image.cols) roi.x = 0;
	if(roi.y >= depth_image.rows) roi.y = 0;

	// Initialise the mask
	cv::Mat_<uchar> mask(depth_image.rows, depth_image.cols, (uchar)0);

	cv::Mat_<uchar> valid_pixels = depth_image > 0;

	// check if there is any depth near the estimate
	if(cv::sum(valid_pixels(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16))/255)[0] > 0)
	{
		double Z = cv::mean(depth_image(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16)), valid_pixels(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16)))[0]; // Z offset from the surface of the face
				
		// Only operate within region of interest of the depth image
		cv::Mat dRoi = depth_image(roi);

		cv::Mat mRoi = mask(roi);

		// Filter all pixels further than 20cm away from the current pose depth estimate
		cv::inRange(dRoi, Z - 200, Z + 200, mRoi);
		
		// Convert to be either 0 or 1
		mask = mask / 255;
		
		cv::Mat_<float> maskF;
		mask.convertTo(maskF, CV_32F);

		//Filter the depth image
		out_depth_image = depth_image.mul(maskF);
	}
	else
	{
		cout << "No depth signal found in foreground, tracking failed" << endl;
		return false;
	}
	return true;
}

// Getting a 3D shape model from the current detected landmarks (in camera space)
cv::Mat_<double> CLNF::GetShape(double fx, double fy, double cx, double cy) const
{
	const int N = mDetectedLandmarks.rows / 2;

	// calculate 3D shape
	cv::Mat_<double> shape3d(N * 3, 1);
	mPDM.CalcShape3D(shape3d, mParamsLocal);
	
	// need to rotate the shape to get the actual 3D representation
	// get the rotation matrix from the euler angles
	const cv::Matx33d R = Euler2RotationMatrix(mParamsGlobal.orient);

	// apply rotation
	shape3d = shape3d.reshape(1, 3);
	shape3d = shape3d.t() * cv::Mat(R).t();
	
	// from the weak perspective model can determine the average depth of the object
	const double Zavg = fx / mParamsGlobal.scale;

	cv::Mat_<double> outShape(N, 3, 0.0);

	// this is described in the paper in section 3.4 (equation 10) (of the CLM-Z paper)
	for (int i = 0; i < N; i++)
	{
		const double Z = Zavg + shape3d.at<double>(i,2);
		const double X = Z * ((mDetectedLandmarks.at<double>(i) - cx)/fx);
		const double Y = Z * ((mDetectedLandmarks.at<double>(i + N) - cy)/fy);

		outShape.at<double>(i,0) = X;
		outShape.at<double>(i,1) = Y;
		outShape.at<double>(i,2) = Z;
	}

	// The format is 3 rows - n cols
	return outShape.t();
}

// A utility bounding box function
cv::Rect_<double> CLNF::GetBoundingBox() const
{
	cv::Mat_<double> xs = mDetectedLandmarks(cv::Rect(0, 0, 1, mDetectedLandmarks.rows/2));
	cv::Mat_<double> ys = mDetectedLandmarks(cv::Rect(0, mDetectedLandmarks.rows/2, 1, mDetectedLandmarks.rows/2));

	double min_x, max_x;
	double min_y, max_y;
	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);

	// See if the detections intersect
	return cv::Rect_<double>(min_x, min_y, max_x - min_x, max_y - min_y);
}

// Getting a head pose estimate from the currently detected landmarks (rotation with respect to point camera)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6d CLNF::GetPoseCamera(double fx, double fy, double cx, double cy)
{
	if (!mDetectedLandmarks.empty() && mParamsGlobal.scale != 0)
	{
		const double Z = fx / mParamsGlobal.scale;
		const double X = ((mParamsGlobal.tx - cx) * (1.0 / fx)) * Z;
		const double Y = ((mParamsGlobal.ty - cy) * (1.0 / fy)) * Z;

		return cv::Vec6d(X, Y, Z, mParamsGlobal.orient[0], mParamsGlobal.orient[1], mParamsGlobal.orient[2]);
	}
	else
	{
		return cv::Vec6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}
}

// Getting a head pose estimate from the currently detected landmarks (rotation in world coordinates)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6d CLNF::GetPoseWorld(double fx, double fy, double cx, double cy)
{
	if (!mDetectedLandmarks.empty() && mParamsGlobal.scale != 0)
	{
		const double Z = fx / mParamsGlobal.scale;
		const double X = ((mParamsGlobal.tx - cx) * (1.0 / fx)) * Z;
		const double Y = ((mParamsGlobal.ty - cy) * (1.0 / fy)) * Z;

		// Here we correct for the camera orientation, for this need to determine the angle the camera makes with the head pose
		double z_x = cv::sqrt(X * X + Z * Z);
		double eul_x = atan2(Y, z_x);

		double z_y = cv::sqrt(Y * Y + Z * Z);
		double eul_y = -atan2(X, z_y);

		cv::Matx33d camera_rotation = Euler2RotationMatrix(cv::Vec3d(eul_x, eul_y, 0));
		cv::Matx33d head_rotation = AxisAngle2RotationMatrix(mParamsGlobal.orient);

		cv::Matx33d corrected_rotation = camera_rotation.t() * head_rotation;

		cv::Vec3d euler_corrected = RotationMatrix2Euler(corrected_rotation);

		return cv::Vec6d(X, Y, Z, euler_corrected[0], euler_corrected[1], euler_corrected[2]);
	}
	else
	{
		return cv::Vec6d(0, 0, 0, 0, 0, 0);
	}
}

// Getting a head pose estimate from the currently detected landmarks, with appropriate correction due to orthographic camera issue
// This is because rotation estimate under orthographic assumption is only correct close to the centre of the image
// This method returns a corrected pose estimate with respect to world coordinates (Experimental)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
void CLNF::GetCorrectedPoseWorld(cv::Vec3d& position, cv::Vec3d& orientation, double fx, double fy, double cx, double cy)
{
	if (!mDetectedLandmarks.empty() && mParamsGlobal.scale != 0.0)
	{
		// This is used as an initial estimate for the iterative PnP algorithm
		const double Z = fx / mParamsGlobal.scale;
		const double X = ((mParamsGlobal.tx - cx) * (1.0 / fx)) * Z;
		const double Y = ((mParamsGlobal.ty - cy) * (1.0 / fy)) * Z;

		// Correction for orientation

		// 2D points
		cv::Mat_<double> landmarks_2D = mDetectedLandmarks.reshape(1, 2).t();

		// 3D points
		cv::Mat_<double> landmarks_3D;
		mPDM.CalcShape3D(landmarks_3D, mParamsLocal);
		landmarks_3D = landmarks_3D.reshape(1, 3).t();

		// The camera matrix
		cv::Matx33d camera_matrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);

		// set position
		position[0] = X;
		position[1] = Y;
		position[2] = Z;

		// set orientation
		orientation = mParamsGlobal.orient;

		// solvePnP
		cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, cv::Mat(), orientation, position, true);

		// convert to euler angles
		orientation = AxisAngle2Euler(orientation);
	}
	else
	{
		position = 0.0;
		orientation = 0.0;
	}
}

// Getting a head pose estimate from the currently detected landmarks, with appropriate correction due to perspective projection
// This method returns a corrected pose estimate with respect to a point camera (NOTE not the world coordinates) (Experimental)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6d CLNF::GetCorrectedPoseCamera(double fx, double fy, double cx, double cy)
{
	if (!mDetectedLandmarks.empty() && mParamsGlobal.scale != 0)
	{
		const double Z = fx / mParamsGlobal.scale;
		const double X = ((mParamsGlobal.tx - cx) * (1.0 / fx)) * Z;
		const double Y = ((mParamsGlobal.ty - cy) * (1.0 / fy)) * Z;

		// Correction for orientation

		// 3D points
		cv::Mat_<double> landmarks_3D;
		mPDM.CalcShape3D(landmarks_3D, mParamsLocal);

		landmarks_3D = landmarks_3D.reshape(1, 3).t();

		// 2D points
		cv::Mat_<double> landmarks_2D = mDetectedLandmarks;

		landmarks_2D = landmarks_2D.reshape(1, 2).t();

		// Solving the PNP model

		// The camera matrix
		cv::Matx33d camera_matrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);

		cv::Vec3d vec_trans(X, Y, Z);
		cv::Vec3d vec_rot(mParamsGlobal.orient);

		cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, cv::Mat(), vec_rot, vec_trans, true);

		// Here we correct for the camera orientation, for this need to determine the angle the camera makes with the head pose
		double z_x = cv::sqrt(vec_trans[0] * vec_trans[0] + vec_trans[2] * vec_trans[2]);
		double eul_x = atan2(vec_trans[1], z_x);

		double z_y = cv::sqrt(vec_trans[1] * vec_trans[1] + vec_trans[2] * vec_trans[2]);
		double eul_y = -atan2(vec_trans[0], z_y);

		cv::Matx33d camera_rotation = Euler2RotationMatrix(cv::Vec3d(eul_x, eul_y, 0));
		cv::Matx33d head_rotation = AxisAngle2RotationMatrix(vec_rot);

		cv::Matx33d corrected_rotation = camera_rotation * head_rotation;

		cv::Vec3d euler_corrected = RotationMatrix2Euler(corrected_rotation);

		return cv::Vec6d(vec_trans[0], vec_trans[1], vec_trans[2], euler_corrected[0], euler_corrected[1], euler_corrected[2]);
	}
	else
	{
		return cv::Vec6d(0, 0, 0, 0, 0, 0);
	}
}

// If landmark detection in video succeeded create a template for use in simple tracking
void CLNF::UpdateTemplate(const cv::Mat_<uchar>& grayscale_image)
{
	// calculate boundingbox
	cv::Rect bounding_box = mPDM.CalcBoundingBox(mParamsGlobal, mParamsLocal);

	// Make sure the box is not out of bounds
	bounding_box = bounding_box & cv::Rect(0, 0, grayscale_image.cols, grayscale_image.rows);

	mFaceTemplate = grayscale_image(bounding_box).clone();
}

// This method uses basic template matching in order to allow for better tracking of fast moving faces
void CLNF::CorrectGlobalParametersVideo(const cv::Mat_<uchar>& grayscale_image, const FaceModelParameters& params)
{
	// calculate boundingbox
	cv::Rect init_box = mPDM.CalcBoundingBox(mParamsGlobal, mParamsLocal);

	cv::Rect roi(
		init_box.x - init_box.width / 2, 
		init_box.y - init_box.height / 2, 
		init_box.width * 2, 
		init_box.height * 2);

	roi = roi & cv::Rect(0, 0, grayscale_image.cols, grayscale_image.rows);

	int off_x = roi.x;
	int off_y = roi.y;

	double scaling = params.face_template_scale / mParamsGlobal.scale;
	cv::Mat_<uchar> image;
	if (scaling < 1.0)
	{
		cv::resize(mFaceTemplate, mFaceTemplate, cv::Size(), scaling, scaling);
		cv::resize(grayscale_image(roi), image, cv::Size(), scaling, scaling);
	}
	else
	{
		scaling = 1.0;
		image = grayscale_image(roi).clone();
	}

	// Resizing the template			
	cv::Mat corr_out;
	cv::matchTemplate(image, mFaceTemplate, corr_out, CV_TM_CCOEFF_NORMED);

	// Actually matching it
	int max_loc[2];
	cv::minMaxIdx(corr_out, NULL, NULL, NULL, max_loc);

	cv::Rect_<double> out_bbox(
		max_loc[1] / scaling + off_x, 
		max_loc[0] / scaling + off_y, 
		mFaceTemplate.rows / scaling, 
		mFaceTemplate.cols / scaling);

	const double shift_x = out_bbox.x - (double)init_box.x;
	const double shift_y = out_bbox.y - (double)init_box.y;

	mParamsGlobal.tx = mParamsGlobal.tx + shift_x;
	mParamsGlobal.ty = mParamsGlobal.ty + shift_y;
}

//===========================================================================
// Visualisation functions
//===========================================================================

void CLNF::DrawBox(cv::Mat& image, cv::Vec3d& position, cv::Vec3d& orientation, const cv::Scalar color, const int thickness, const float fx, const float fy, const float cx, const float cy)
{
	double boxVerts[] = 
	{	
		-1.0,  1.0, -1.0,
		 1.0,  1.0, -1.0,
		 1.0,  1.0,  1.0,
		-1.0,  1.0,  1.0,
		 1.0, -1.0,  1.0,
		 1.0, -1.0, -1.0,
		-1.0, -1.0, -1.0,
		-1.0, -1.0,  1.0 
	};

	const vector<pair<int, int>> edges = {
		pair<int, int>(0, 1),
		pair<int, int>(1, 2),
		pair<int, int>(2, 3),
		pair<int, int>(0, 3),
		pair<int, int>(2, 4),
		pair<int, int>(1, 5),
		pair<int, int>(0, 6),
		pair<int, int>(3, 7),
		pair<int, int>(6, 5),
		pair<int, int>(5, 4),
		pair<int, int>(4, 7),
		pair<int, int>(7, 6)
	};

	// The size of the head is roughly 200mm x 200mm x 200mm
	cv::Mat_<double> box = cv::Mat(8, 3, CV_64F, boxVerts).clone() * 100;

	// Get rotation
	cv::Matx33d rot = Euler2RotationMatrix(orientation);
	
	// Rotate the box
	cv::Mat_<double> rotBox;
	rotBox = cv::Mat(rot) * box.t();
	rotBox = rotBox.t();

	// Move the bounding box to head position
	rotBox.col(0) = rotBox.col(0) + position[0];
	rotBox.col(1) = rotBox.col(1) + position[1];
	rotBox.col(2) = rotBox.col(2) + position[2];

	// Project the box
	cv::Mat_<double> rotBoxProj;
	Project(rotBoxProj, rotBox, fx, fy, cx, cy);

	cv::Rect image_rect(0, 0, image.cols, image.rows);

	// draw the lines
	for (size_t i = 0; i < edges.size(); ++i)
	{
		cv::Mat_<double> begin;
		cv::Mat_<double> end;

		rotBoxProj.row(edges[i].first).copyTo(begin);
		rotBoxProj.row(edges[i].second).copyTo(end);

		const cv::Point p1((int)begin.at<double>(0), (int)begin.at<double>(1));
		const cv::Point p2((int)end.at<double>(0), (int)end.at<double>(1));

		// Only draw the line if one of the points is inside the image
		if (p1.inside(image_rect) || p2.inside(image_rect))
			cv::line(image, p1, p2, color, thickness);
	}
}

// Computing landmarks (to be drawn later possibly)
vector<cv::Point2d> CLNF::CalculateLandmarks(const cv::Mat_<double>& shape2D, cv::Mat_<int>& visibilities)
{
	int n = shape2D.rows / 2;
	vector<cv::Point2d> landmarks;

	for (int i = 0; i < n; ++i)
	{
		if (visibilities.at<int>(i))
		{
			cv::Point2d featurePoint(shape2D.at<double>(i), shape2D.at<double>(i + n));

			landmarks.push_back(featurePoint);
		}
	}

	return landmarks;
}

// Computing landmarks (to be drawn later possibly)
vector<cv::Point2d> CLNF::CalculateLandmarks(cv::Mat img, const cv::Mat_<double>& shape2D)
{
	int n;
	vector<cv::Point2d> landmarks;

	if (shape2D.cols == 2)
	{
		n = shape2D.rows;
	}
	else if (shape2D.cols == 1)
	{
		n = shape2D.rows / 2;
	}

	for (int i = 0; i < n; ++i)
	{
		cv::Point2d featurePoint;
		if (shape2D.cols == 1)
		{
			featurePoint = cv::Point2d(shape2D.at<double>(i), shape2D.at<double>(i + n));
		}
		else
		{
			featurePoint = cv::Point2d(shape2D.at<double>(i, 0), shape2D.at<double>(i, 1));
		}

		landmarks.push_back(featurePoint);
	}

	return landmarks;
}

// Computing landmarks (to be drawn later possibly)
vector<cv::Point2d> CLNF::CalculateLandmarks()
{
	const size_t idx = mPatchExperts.getViewIdx(mParamsGlobal, 0);

	// get reference to visibilities
	const vector<vector<cv::Mat_<int>>>& VISIBILITIES =
		mPatchExperts.getVisibilities();

	// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
	return CalculateLandmarks(mDetectedLandmarks, VISIBILITIES[0][idx]);
}

// Drawing landmarks on a face image
void CLNF::Draw(cv::Mat& img, const cv::Mat_<double>& shape2D, const cv::Mat_<int>& visibilities)
{
	const int n = shape2D.rows / 2;

	// main model
	if (n >= 66)
	{
		for (int i = 0; i < n; ++i)
		{
			if (visibilities.at<int>(i))
			{
				cv::Point featurePoint(
					(int)shape2D.at<double>(i), 
					(int)shape2D.at<double>(i + n));

				// A rough heuristic for drawn point size
				int thickness = (int)std::ceil(3.0* ((double)img.cols) / 640.0);
				int thickness_2 = (int)std::ceil(1.0* ((double)img.cols) / 640.0);

				cv::circle(img, featurePoint, 1, cv::Scalar(0, 0, 255), thickness);
				cv::circle(img, featurePoint, 1, cv::Scalar(255, 0, 0), thickness_2);
			}
		}
	}

	// eye-models
	else if (n == 28) 
	{
		for (int i = 0; i < n; ++i)
		{
			cv::Point featurePoint(
				(int)shape2D.at<double>(i), 
				(int)shape2D.at<double>(i + n));

			int next_point = i + 1;

			if (i == 7)       next_point = 0;
			else if (i == 19) next_point = 8;
			else if (i == 27) next_point = 20;

			cv::Point nextFeaturePoint(
				(int)shape2D.at<double>(next_point), 
				(int)shape2D.at<double>(next_point + n));

			// draw line between points
			if (i < 8 || i > 19)
				cv::line(img, featurePoint, nextFeaturePoint, cv::Scalar(255, 0, 0), 1);
			else
				cv::line(img, featurePoint, nextFeaturePoint, cv::Scalar(0, 0, 255), 1);
		}
	}

	// not used?
	else if (n == 6)
	{
		for (int i = 0; i < n; ++i)
		{
			// first point
			cv::Point featurePoint(
				(int)shape2D.at<double>(i), 
				(int)shape2D.at<double>(i + n));

			int next_point = i + 1;

			if (i == 5)
				next_point = 0;

			// second point
			cv::Point nextFeaturePoint(
				(int)shape2D.at<double>(next_point), 
				(int)shape2D.at<double>(next_point + n));

			// draw line between points
			cv::line(img, featurePoint, nextFeaturePoint, cv::Scalar(255, 0, 0), 1);
		}
	}

	// inner model has n=51
	else
	{
		// HUH?
	}
}

// Drawing detected landmarks on a face image
void CLNF::Draw(cv::Mat& img)
{
	const size_t IDX = mPatchExperts.getViewIdx(mParamsGlobal, 0);

	// get reference to visibilities
	const vector<vector<cv::Mat_<int>>>& VISIBILITIES =
		mPatchExperts.getVisibilities();

	// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
	Draw(img, mDetectedLandmarks, VISIBILITIES[0][IDX]);
}


cv::Point3f GetPupilPosition(const cv::Mat_<double>& eyeLdmks3d) 
{
	cv::Mat_<double> irisLdmks3d = eyeLdmks3d.t();

	irisLdmks3d = irisLdmks3d.rowRange(0, 8);

	return cv::Point3f(
		mean(irisLdmks3d.col(0))[0], 
		mean(irisLdmks3d.col(1))[0], 
		mean(irisLdmks3d.col(2))[0]);
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

ModelMain::ModelMain(
	const string&         mainPdmFile,
	const vector<string>& mainCcnfFiles,
	const string&         innerPdmFile,
	const vector<string>& innerCcnfFiles,
	const string&         leftEyePdmFile,
	const vector<string>& leftEyeCcnfFiles,
	const string&         rightEyePdmFile,
	const vector<string>& rightEyeCcnfFiles,
	const string&         mainTriangulationsFile) : 
	CLNF(mainPdmFile, mainCcnfFiles, mainTriangulationsFile, Type::Main),
	mInnerModel(innerPdmFile, innerCcnfFiles, "", Type::Inner),
	mLeftEyeModel(leftEyePdmFile, leftEyeCcnfFiles, "", Type::LeftEye),
	mRightEyeModel(rightEyePdmFile, rightEyeCcnfFiles, "", Type::RightEye)
{
	// inner paramters
	vector<int> inner_windows_large;
	vector<int> inner_windows_small;
	inner_windows_large.push_back(9);
	inner_windows_small.push_back(9);
	mInnerParams.window_sizes_init = inner_windows_large;
	mInnerParams.window_sizes_small = inner_windows_small;
	mInnerParams.window_sizes_current = inner_windows_large;
	mInnerParams.reg_factor = 2.5;
	mInnerParams.sigma = 1.75;
	mInnerParams.weight_factor = 2.5;
	mInnerParams.validate_detections = false;
	mInnerParams.refine_parameters = false;

	// left eye paramters
	vector<int> leye_windows_large;
	vector<int> leye_windows_small;
	leye_windows_large.push_back(3);
	leye_windows_large.push_back(5);
	leye_windows_large.push_back(9);
	leye_windows_small.push_back(3);
	leye_windows_small.push_back(5);
	leye_windows_small.push_back(9);
	mLeftEyeParams.window_sizes_init = leye_windows_large;
	mLeftEyeParams.window_sizes_small = leye_windows_small;
	mLeftEyeParams.window_sizes_current = leye_windows_large;
	mLeftEyeParams.reg_factor = 0.5;
	mLeftEyeParams.sigma = 1.0;
	mLeftEyeParams.validate_detections = false;
	mLeftEyeParams.refine_parameters = false;

	// right eye paramters
	vector<int> reye_windows_large;
	vector<int> reye_windows_small;
	reye_windows_large.push_back(3);
	reye_windows_large.push_back(5);
	reye_windows_large.push_back(9);
	reye_windows_small.push_back(3);
	reye_windows_small.push_back(5);
	reye_windows_small.push_back(9);
	mRightEyeParams.window_sizes_init = reye_windows_large;
	mRightEyeParams.window_sizes_small = reye_windows_small;
	mRightEyeParams.window_sizes_current = reye_windows_large;
	mRightEyeParams.reg_factor = 0.5;
	mRightEyeParams.sigma = 1.0;
	mRightEyeParams.validate_detections = false;
	mRightEyeParams.refine_parameters = false;
}

void ModelMain::Draw(cv::Mat& img)
{
	// draw main
	CLNF::Draw(img);

	// draw inner
	if (mInnerModel.getPDM().getNumberOfPoints() != (int)INNER_MAPPING.size())
		mInnerModel.Draw(img);

	// draw left eye
	if (mLeftEyeModel.getPDM().getNumberOfPoints() != (int)LEFT_EYE_MAPPING.size())
		mLeftEyeModel.Draw(img);

	// draw right eye
	if (mRightEyeModel.getPDM().getNumberOfPoints() != (int)RIGHT_EYE_MAPPING.size())
		mRightEyeModel.Draw(img);
}

bool ModelMain::DetectLandmarks(const cv::Mat_<uchar>& image, const cv::Mat_<float>& depth, FaceModelParameters& params, DetectionValidator* validator)
{
	// Fits from the current estimate of local and global parameters in the model
	const bool FIT_SUCCESS = Fit(image, depth, params.window_sizes_current, params);
	
	// Store the landmarks converged on in detected_landmarks
	calcShape2D();

	//////////////////////////////////////////////////////////////////////
	// Process hierarchical models in parallel
	bool parts_used = false;
	tbb::parallel_for(0, (int)3, [&](int part_model)
	{
		FaceModelParameters*          partParams;
		CLNF*                         partModel;
		const vector<pair<int, int>>* partMappings;

		// select inner model to process
		switch (part_model)
		{
		case 0:
			partParams = &mInnerParams;
			partModel = &mInnerModel;
			partMappings = &INNER_MAPPING;
			break;
		
		case 1:
			partParams = &mLeftEyeParams;
			partModel = &mLeftEyeModel;
			partMappings = &LEFT_EYE_MAPPING;
			break;
		
		case 2:
			partParams = &mRightEyeParams;
			partModel = &mRightEyeModel;
			partMappings = &RIGHT_EYE_MAPPING;
			break;

		default:
			assert(1 == 2);
			break;
		}
		
		PDM& partPDM = partModel->getPDM();
			
		const int NUM_MAIN_POINTS = mPDM.getNumberOfPoints();
		const int NUM_PART_POINTS = partPDM.getNumberOfPoints();
			
		cv::Mat_<double> partModelLocs(NUM_PART_POINTS * 2, 1, 0.0);

		// Extract the corresponding landmarks
		for (size_t i = 0; i < partMappings->size(); ++i)
		{
			partModelLocs.at<double>(partMappings->at(i).second) =
				mDetectedLandmarks.at<double>(partMappings->at(i).first);

			partModelLocs.at<double>(partMappings->at(i).second + NUM_PART_POINTS) =
				mDetectedLandmarks.at<double>(partMappings->at(i).first + NUM_MAIN_POINTS);
		}

		// Fit the part based model PDM
		partPDM.CalcParams(
			partModel->getParamsGlobal(),
			partModel->getParamsLocal(),
			partModelLocs);

		// Only do this if we don't need to upsample
		if (mParamsGlobal.scale > 0.9 * partModel->getPatchExperts().getPatchScaling()[0])
		{
			parts_used = true;

			partParams->window_sizes_current = 
				partParams->window_sizes_init;

			// Do the actual landmark detection, 
			// hierarchical models have no validator
			partModel->DetectLandmarks(image, depth, *partParams);
		}
		else
		{
			partModel->calcShape2D();
		}
	});

	// Recompute main model based on the fit part models
	if (parts_used)
	{
		// Reincorporate the inner model into main tracker
		const cv::Mat_<double>& landmarksInner = mInnerModel.getDetectedLandmarks();
		for (size_t mapping_ind = 0; mapping_ind < INNER_MAPPING.size(); ++mapping_ind)
		{
			mDetectedLandmarks.at<double>(INNER_MAPPING[mapping_ind].first) =
				landmarksInner.at<double>(INNER_MAPPING[mapping_ind].second);

			mDetectedLandmarks.at<double>(INNER_MAPPING[mapping_ind].first + mPDM.getNumberOfPoints()) =
				landmarksInner.at<double>(INNER_MAPPING[mapping_ind].second + mInnerModel.getPDM().getNumberOfPoints());
		}

		// Reincorporate the left-eye model into main tracker
		const cv::Mat_<double>& landmarksLeftEye = mLeftEyeModel.getDetectedLandmarks();
		for (size_t mapping_ind = 0; mapping_ind < LEFT_EYE_MAPPING.size(); ++mapping_ind)
		{
			mDetectedLandmarks.at<double>(LEFT_EYE_MAPPING[mapping_ind].first) =
				landmarksLeftEye.at<double>(LEFT_EYE_MAPPING[mapping_ind].second);

			mDetectedLandmarks.at<double>(LEFT_EYE_MAPPING[mapping_ind].first + mPDM.getNumberOfPoints()) =
				landmarksLeftEye.at<double>(LEFT_EYE_MAPPING[mapping_ind].second + mLeftEyeModel.getPDM().getNumberOfPoints());
		}

		// Reincorporate the right-eye model into main tracker
		const cv::Mat_<double>& landmarksRightEye = mRightEyeModel.getDetectedLandmarks();
		for (size_t mapping_ind = 0; mapping_ind < RIGHT_EYE_MAPPING.size(); ++mapping_ind)
		{
			mDetectedLandmarks.at<double>(RIGHT_EYE_MAPPING[mapping_ind].first) =
				landmarksRightEye.at<double>(RIGHT_EYE_MAPPING[mapping_ind].second);

			mDetectedLandmarks.at<double>(RIGHT_EYE_MAPPING[mapping_ind].first + mPDM.getNumberOfPoints()) =
				landmarksRightEye.at<double>(RIGHT_EYE_MAPPING[mapping_ind].second + mRightEyeModel.getPDM().getNumberOfPoints());
		}

		///

		calcParams();
		calcShape2D();
	}
	
	//////////////////////////////////////////////////////////////////////
	// Check detection correctness using a CNN validator
	if (validator != nullptr && params.validate_detections && FIT_SUCCESS)
	{
		mDetectionCertainty = validator->Check(mParamsGlobal.orient, image, mDetectedLandmarks);
		mDetectionSuccess = mDetectionCertainty < params.validation_boundary;
	}
	else
	{
		mDetectionSuccess = FIT_SUCCESS;
		mDetectionCertainty = FIT_SUCCESS ? -1.0 : 1.0;
	}

	return mDetectionSuccess;
}


bool ModelMain::DetectLandmarksInVideo(
	FaceDetectorDlib&      face_detector,
	const cv::Mat_<uchar>& grayscale_image,
	const cv::Mat_<float>& depth_image,
	FaceModelParameters&   params,
	DetectionValidator*    validator)
{
	// First need to decide if the landmarks should be "detected" or "tracked"
	// Detected means running face detection and a larger search area, 
	// tracked means initialising from previous step and using a smaller search area

	// Indicating that this is a first detection in video sequence or after restart
	const bool INIT_DETECT = !mTrackingInitialised;

	// Only do it if there was a face detection at all
	if (mTrackingInitialised)
	{
		// The area of interest search size will depend,
		// if the previous track was successful
		params.window_sizes_current = !mDetectionSuccess ?
			params.window_sizes_init : params.window_sizes_small;

		// Before the expensive landmark detection step apply a quick template tracking approach
		if (params.use_face_template && !mFaceTemplate.empty() && mDetectionSuccess)
			CorrectGlobalParametersVideo(grayscale_image, params);

		// Perform the actual Landmark Detection
		const bool TRACK_SUCCESS = DetectLandmarks(grayscale_image, depth_image, params, validator);

		// Make a record that tracking failed
		if (!TRACK_SUCCESS)
		{
			mFailuresInARow++;
		}
		else
		{
			// indicate that tracking is a success
			mFailuresInARow = -1;
			UpdateTemplate(grayscale_image);
		}
	}

	/////////////////////////////////////////////////////////////////////////////

	// This is used for both detection (if it the tracking has not been initialised yet) 
	// or if the tracking failed (however we do this every n frames, for speed)
	// This also has the effect of an attempt to reinitialise just after 
	// the tracking has failed, which is useful during large motions

	const bool COND1 =
		!mTrackingInitialised &&
		((mFailuresInARow + 1) % (params.reinit_video_every * 6) == 0);

	const bool COND2 =
		mTrackingInitialised &&
		!mDetectionSuccess &&
		params.reinit_video_every > 0 &&
		(mFailuresInARow % params.reinit_video_every == 0);

	if (COND1 || COND2)
	{
		// returns from face detect
		cv::Rect_<double> bounding_box;
		double            confidence;

		// perform face detection
		const bool FACE_DETECT_SUCCESS = face_detector.DetectSingleFace(
			bounding_box, grayscale_image, confidence);

		// Attempt to detect landmarks using the detected face 
		// (if unseccessful the detection will be ignored)
		if (FACE_DETECT_SUCCESS)
		{
			// Indicate that tracking has started as a face was detected
			mTrackingInitialised = true;

			// Keep track of old model values so that they can be restored if redetection fails
			GlobalParameters params_global_init = mParamsGlobal;
			cv::Mat_<double> params_local_init = mParamsLocal.clone();

			double likelihood_init = mModelLikelihood;
			cv::Mat_<double> detected_landmarks_init = mDetectedLandmarks.clone();
			cv::Mat_<double> landmark_likelihoods_init = mLandmarkLikelihoods.clone();

			// Use the detected bounding box and empty local parameters
			mParamsLocal.setTo(0);
			mPDM.CalcParams(mParamsGlobal, bounding_box, mParamsLocal);

			// Make sure the search size is large
			params.window_sizes_current = params.window_sizes_init;

			// Do the actual landmark detection (and keep it only if successful)
			const bool LANDMARK_DETECT_SUCCESS = DetectLandmarks(
				grayscale_image, depth_image, params, validator);

			// If landmark reinitialisation unsucessful continue from previous estimates
			// if it's initial detection however, do not care if it was successful as 
			// the validator might be wrong, so continue trackig regardless
			if (!INIT_DETECT && !LANDMARK_DETECT_SUCCESS)
			{
				// Restore previous estimates
				mParamsGlobal = params_global_init;
				mParamsLocal = params_local_init.clone();

				mPDM.CalcShape2D(mDetectedLandmarks, mParamsLocal, mParamsGlobal);

				mModelLikelihood = likelihood_init;
				mDetectedLandmarks = detected_landmarks_init.clone();
				mLandmarkLikelihoods = landmark_likelihoods_init.clone();

				return false;
			}
			else
			{
				mFailuresInARow = -1;
				UpdateTemplate(grayscale_image);
				return true;
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////

	// if the model has not been initialised yet class it as a failure
	if (!mTrackingInitialised)
		mFailuresInARow++;

	// un-initialise the tracking
	if (mFailuresInARow > 100)
		mTrackingInitialised = false;

	return mDetectionSuccess;
}

bool ModelMain::DetectLandmarksInVideo(FaceDetectorDlib& face_detector, const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, const cv::Rect_<double> bounding_box, FaceModelParameters& params, DetectionValidator* validator)
{
	if (bounding_box.width > 0)
	{
		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		mParamsLocal.setTo(0);
		mPDM.CalcParams(mParamsGlobal, bounding_box, mParamsLocal);

		// indicate that face was detected so initialisation is not necessary
		mTrackingInitialised = true;
	}

	return DetectLandmarksInVideo(face_detector, grayscale_image, depth_image, params, validator);
}

bool ModelMain::DetectLandmarksInVideo(FaceDetectorDlib& face_detector, const cv::Mat_<uchar> &grayscale_image, FaceModelParameters& params, DetectionValidator* validator)
{
	return DetectLandmarksInVideo(face_detector, grayscale_image, cv::Mat_<float>(), params, validator);
}

bool ModelMain::DetectLandmarksInVideo(FaceDetectorDlib& face_detector, const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, FaceModelParameters& params, DetectionValidator* validator)
{
	return DetectLandmarksInVideo(face_detector, grayscale_image, cv::Mat_<float>(), params, validator);
}

void ModelMain::DrawGaze(cv::Mat& img, Result& result, float fx, float fy, float cx, float cy)
{
	cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);

	// shapes
	cv::Mat eyeLdmks3d_left = mLeftEyeModel.GetShape(fx, fy, cx, cy);
	cv::Mat eyeLdmks3d_right = mRightEyeModel.GetShape(fx, fy, cx, cy);
	
	// pupil positions
	cv::Point3f pupil_left = GetPupilPosition(eyeLdmks3d_left);
	cv::Point3f pupil_right = GetPupilPosition(eyeLdmks3d_right);

	vector<cv::Point3d> points_left;
	points_left.push_back(cv::Point3d(pupil_left));
	points_left.push_back(cv::Point3d(pupil_left + ::cv::Point3f(result.eyeLeft.gaze * 50.0)));

	vector<cv::Point3d> points_right;
	points_right.push_back(cv::Point3d(pupil_right));
	points_right.push_back(cv::Point3d(pupil_right + ::cv::Point3f(result.eyeRight.gaze * 50.0)));

	cv::Mat_<double> proj_points;
	
	// left
	
	cv::Mat_<double> mesh_0 = (cv::Mat_<double>(2, 3) << 
		points_left[0].x, points_left[0].y, points_left[0].z, 
		points_left[1].x, points_left[1].y, points_left[1].z);
	
	Project(proj_points, mesh_0, fx, fy, cx, cy);
	
	cv::line(img, 
		cv::Point(proj_points.at<double>(0, 0), proj_points.at<double>(0, 1)), 
		cv::Point(proj_points.at<double>(1, 0), proj_points.at<double>(1, 1)), 
		cv::Scalar(110, 220, 0), 2, 8);

	// right

	cv::Mat_<double> mesh_1 = (cv::Mat_<double>(2, 3) << 
		points_right[0].x, points_right[0].y, points_right[0].z, 
		points_right[1].x, points_right[1].y, points_right[1].z);
	
	Project(proj_points, mesh_1, fx, fy, cx, cy);
	
	cv::line(img, 
		cv::Point(proj_points.at<double>(0, 0), proj_points.at<double>(0, 1)), 
		cv::Point(proj_points.at<double>(1, 0), proj_points.at<double>(1, 1)), 
		cv::Scalar(110, 220, 0), 2, 8);
}

void ModelMain::EstimateGaze(Result& result, float fx, float fy, float cx, float cy, bool left_eye)
{
	cv::Vec6d headPose = GetPoseCamera(fx, fy, cx, cy);
	cv::Vec3d eulerAngles(headPose(3), headPose(4), headPose(5));
	cv::Matx33d rotMat = Euler2RotationMatrix(eulerAngles);

	cv::Mat eyeLdmks3d = left_eye ? 
		mLeftEyeModel.GetShape(fx, fy, cx, cy) : 
		mRightEyeModel.GetShape(fx, fy, cx, cy);

	cv::Point3f pupil = GetPupilPosition(eyeLdmks3d);
	cv::Point3f rayDir = pupil / norm(pupil);

	cv::Mat faceLdmks3d = GetShape(fx, fy, cx, cy);
	faceLdmks3d = faceLdmks3d.t();
	cv::Mat offset = (cv::Mat_<double>(3, 1) << 0, -3.50, 0);
	
	const int EYEIDX = left_eye ?  0 : 1;
	
	cv::Mat eyeballCentreMat = (faceLdmks3d.row(36 + EYEIDX * 6) + faceLdmks3d.row(39 + EYEIDX * 6)) / 2.0f + (cv::Mat(rotMat)*offset).t();
	cv::Point3f eyeballCentre = cv::Point3f(eyeballCentreMat);
	cv::Point3f gazeVecAxis = RaySphereIntersect(cv::Point3f(0, 0, 0), rayDir, eyeballCentre, 12) - eyeballCentre;
	cv::Point3f p = gazeVecAxis / norm(gazeVecAxis);
	
	// attention is delta angle
	const float ANGLE = acos(p.dot(-rayDir));
	const float MINATT = M_PI * 0.1f; // 18° = 0
	const float CONV = 1.0f / MINATT;
	const float MAPPED = fmax(0.0f, fmin(1.0f, 1.1f - (ANGLE * CONV)));

	if (left_eye)
	{		
		result.eyeLeft.gaze = cv::Vec3f(p);
		result.eyeLeft.position = ::cv::Vec3f(pupil);
		result.eyeLeft.attention = MAPPED;
	}
	else
	{
		result.eyeRight.gaze = cv::Vec3f(p);
		result.eyeRight.position = ::cv::Vec3f(pupil);
		result.eyeRight.attention = MAPPED;
	}
}