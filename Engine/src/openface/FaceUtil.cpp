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

using namespace std;

namespace OpenFace
{

	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================

	void crossCorr_m(const cv::Mat_<float>& img, cv::Mat_<double>& img_dft, const cv::Mat_<float>& _templ, map<int, cv::Mat_<double> >& _templ_dfts, cv::Mat_<float>& corr)
	{
		// Our model will always be under min block size so can ignore this
		//const double blockScale = 4.5;
		//const int minBlockSize = 256;

		int maxDepth = CV_64F;

		cv::Size dftsize;

		dftsize.width = cv::getOptimalDFTSize(corr.cols + _templ.cols - 1);
		dftsize.height = cv::getOptimalDFTSize(corr.rows + _templ.rows - 1);

		// Compute block size
		cv::Size blocksize;
		blocksize.width = dftsize.width - _templ.cols + 1;
		blocksize.width = MIN(blocksize.width, corr.cols);
		blocksize.height = dftsize.height - _templ.rows + 1;
		blocksize.height = MIN(blocksize.height, corr.rows);

		cv::Mat_<double> dftTempl;

		// if this has not been precomputed, precompute it, otherwise use it
		if (_templ_dfts.find(dftsize.width) == _templ_dfts.end())
		{
			dftTempl.create(dftsize.height, dftsize.width);

			cv::Mat_<float> src = _templ;

			cv::Mat_<double> dst(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));

			cv::Mat_<double> dst1(dftTempl, cv::Rect(0, 0, _templ.cols, _templ.rows));

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			if (dst.cols > _templ.cols)
			{
				cv::Mat_<double> part(dst, cv::Range(0, _templ.rows), cv::Range(_templ.cols, dst.cols));
				part.setTo(0);
			}

			// Perform DFT of the template
			dft(dst, dst, 0, _templ.rows);

			_templ_dfts[dftsize.width] = dftTempl;

		}
		else
		{
			// use the precomputed version
			dftTempl = _templ_dfts.find(dftsize.width)->second;
		}

		cv::Size bsz(std::min(blocksize.width, corr.cols), std::min(blocksize.height, corr.rows));
		cv::Mat src;

		cv::Mat cdst(corr, cv::Rect(0, 0, bsz.width, bsz.height));

		cv::Mat_<double> dftImg;

		if (img_dft.empty())
		{
			dftImg.create(dftsize);
			dftImg.setTo(0.0);

			cv::Size dsz(bsz.width + _templ.cols - 1, bsz.height + _templ.rows - 1);

			int x2 = std::min(img.cols, dsz.width);
			int y2 = std::min(img.rows, dsz.height);

			cv::Mat src0(img, cv::Range(0, y2), cv::Range(0, x2));
			cv::Mat dst(dftImg, cv::Rect(0, 0, dsz.width, dsz.height));
			cv::Mat dst1(dftImg, cv::Rect(0, 0, x2, y2));

			src = src0;

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			dft(dftImg, dftImg, 0, dsz.height);
			img_dft = dftImg.clone();
		}

		cv::Mat dftTempl1(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
		cv::mulSpectrums(img_dft, dftTempl1, dftImg, 0, true);
		cv::dft(dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height);

		src = dftImg(cv::Rect(0, 0, bsz.width, bsz.height));

		src.convertTo(cdst, CV_32F);

	}

	void matchTemplate_m(const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, map<int, cv::Mat_<double> >& templ_dfts, cv::Mat_<float>& result, int method)
	{
		int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 : method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
		bool isNormed = method == CV_TM_CCORR_NORMED || method == CV_TM_SQDIFF_NORMED || method == CV_TM_CCOEFF_NORMED;

		// Assume result is defined properly
		if (result.empty())
		{
			cv::Size corrSize(input_img.cols - templ.cols + 1, input_img.rows - templ.rows + 1);
			result.create(corrSize);
		}

		crossCorr_m(input_img, img_dft, templ, templ_dfts, result);

		if (method == CV_TM_CCORR)
			return;

		double invArea = 1. / ((double)templ.rows * templ.cols);

		cv::Mat sum, sqsum;
		cv::Scalar templMean, templSdv;
		double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
		double templNorm = 0, templSum2 = 0;

		if (method == CV_TM_CCOEFF)
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, CV_64F);
			}
			sum = _integral_img;

			templMean = cv::mean(templ);
		}
		else
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, _integral_img_sq, CV_64F);
			}

			sum = _integral_img;
			sqsum = _integral_img_sq;

			meanStdDev(templ, templMean, templSdv);

			templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

			if (templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED)
			{
				result.setTo(1.0);
				return;
			}

			templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];

			if (numType != 1)
			{
				templMean = cv::Scalar::all(0);
				templNorm = templSum2;
			}

			templSum2 /= invArea;
			templNorm = std::sqrt(templNorm);
			templNorm /= std::sqrt(invArea); // care of accuracy here

			q0 = (double*)sqsum.data;
			q1 = q0 + templ.cols;
			q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
			q3 = q2 + templ.cols;
		}

		double* p0 = (double*)sum.data;
		double* p1 = p0 + templ.cols;
		double* p2 = (double*)(sum.data + templ.rows*sum.step);
		double* p3 = p2 + templ.cols;

		int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
		int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

		int i, j;

		for (i = 0; i < result.rows; i++)
		{
			float* rrow = result.ptr<float>(i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for (j = 0; j < result.cols; j++, idx += 1, idx2 += 1)
			{
				double num = rrow[j], t;
				double wndMean2 = 0, wndSum2 = 0;

				if (numType == 1)
				{

					t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
					wndMean2 += t*t;
					num -= t*templMean[0];

					wndMean2 *= invArea;
				}

				if (isNormed || numType == 2)
				{

					t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
					wndSum2 += t;

					if (numType == 2)
					{
						num = wndSum2 - 2 * num + templSum2;
						num = MAX(num, 0.);
					}
				}

				if (isNormed)
				{
					t = std::sqrt(MAX(wndSum2 - wndMean2, 0))*templNorm;
					if (fabs(num) < t)
						num /= t;
					else if (fabs(num) < t*1.125)
						num = num > 0 ? 1 : -1;
					else
						num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
				}

				rrow[j] = (float)num;
			}
		}
	}

}
