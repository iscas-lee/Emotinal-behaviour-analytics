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

void SVMDynamicLinear::Read(std::istream& stream, const ActionUnit& au_name)
{
	// read the means from the first one
	if (mMeans.empty())
		ReadMatBin(stream, mMeans);
	
	// otherwise just check if they're equal (supposed to!)
	else
	{
		cv::Mat_<double> m_tmp;
		ReadMatBin(stream, m_tmp);

		if(cv::norm(m_tmp - mMeans > 0.00001))		
			cout << "Something went wrong with the SVM dynamic classifiers" << endl;		
	}

	cv::Mat_<double> support_vectors_curr;
	ReadMatBin(stream, support_vectors_curr);

	double bias;
	double pos_class;
	double neg_class;

	stream.read((char*)&bias, 8);
	stream.read((char*)&pos_class, 8);
	stream.read((char*)&neg_class, 8);

	// Add a column vector to the matrix of support vectors (each column is a support vector)
	if(!mSupportVectors.empty())
	{	
		cv::transpose(mSupportVectors, mSupportVectors);
		cv::transpose(support_vectors_curr, support_vectors_curr);

		mSupportVectors.push_back(support_vectors_curr);
		cv::transpose(mSupportVectors, mSupportVectors);

		cv::transpose(mBiases, mBiases);
		mBiases.push_back(cv::Mat_<double>(1, 1, bias));
		cv::transpose(mBiases, mBiases);
	}
	else
	{
		mSupportVectors.push_back(support_vectors_curr);
		mBiases.push_back(cv::Mat_<double>(1, 1, bias));
	}

	mPosClasses.push_back(pos_class);
	mNegClasses.push_back(neg_class);
	mAUNames.push_back(au_name);
}

// Prediction using the HOG descriptor
void SVMDynamicLinear::Predict(
	ActionUnitValues&       predictions, 
	const cv::Mat_<double>& fhog_descriptor, 
	const cv::Mat_<double>& geom_params,  
	const cv::Mat_<double>& running_median,  
	const cv::Mat_<double>& running_median_geom)
{
	if (mAUNames.size() <= 0)
		return;

	// stores prediction calc result
	cv::Mat_<double> preds;

	if (fhog_descriptor.cols == mMeans.cols)
	{
		// calculate predictions
		preds = (fhog_descriptor - mMeans - running_median) * mSupportVectors + mBiases;
	}
	else
	{
		cv::Mat_<double> input;
		cv::Mat_<double> run_med;
			
		cv::hconcat(fhog_descriptor, geom_params, input);
		cv::hconcat(running_median, running_median_geom, run_med);

		// calculate predictions
		preds = (input - mMeans - run_med) * mSupportVectors + mBiases;
	}

	// write predictions to output struct
	for (int i = 0; i < preds.cols; ++i)
	{
		const bool POSITIVE = preds.at<double>(i) > 0;
		const double VALUE  = POSITIVE ? mPosClasses[i] : mNegClasses[i];

		switch (mAUNames[i])
		{
		case ActionUnit::AU01: predictions.AU01 = VALUE; break;
		case ActionUnit::AU02: predictions.AU02 = VALUE; break;
		case ActionUnit::AU04: predictions.AU04 = VALUE; break;
		case ActionUnit::AU05: predictions.AU05 = VALUE; break;
		case ActionUnit::AU06: predictions.AU06 = VALUE; break;
		case ActionUnit::AU07: predictions.AU07 = VALUE; break;
		case ActionUnit::AU09: predictions.AU09 = VALUE; break;
		case ActionUnit::AU10: predictions.AU10 = VALUE; break;
		case ActionUnit::AU12: predictions.AU12 = VALUE; break;
		case ActionUnit::AU14: predictions.AU14 = VALUE; break;
		case ActionUnit::AU15: predictions.AU15 = VALUE; break;
		case ActionUnit::AU17: predictions.AU17 = VALUE; break;
		case ActionUnit::AU20: predictions.AU20 = VALUE; break;
		case ActionUnit::AU23: predictions.AU23 = VALUE; break;
		case ActionUnit::AU25: predictions.AU25 = VALUE; break;
		case ActionUnit::AU26: predictions.AU26 = VALUE; break;
		case ActionUnit::AU45: predictions.AU45 = VALUE; break;
		}
	}
}