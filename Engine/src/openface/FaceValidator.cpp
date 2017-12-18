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
// Read in the landmark detection validation module
DetectionValidator::DetectionValidator(const string location)
{
	ifstream stream (location, ios::in | ios::binary);

	if (stream.is_open())	
	{				
		stream.seekg(0, ios::beg);

		///////////////////////////////////////
		// Read validator type
		int type;
		stream.read((char*)&type, 4);
		mValidatorType = (Type)type;

		// SVR/NN not supported anymore
		assert(mValidatorType == Type::CNN);

		///////////////////////////////////////
		// Read the number of views (orientations)
		int n;
		stream.read((char*)&n, 4);
	
		// resize everything to number of views
		mOrientations.resize(n);
		mPaws.resize(n);
		mMeanImages.resize(n);
		mStandardDeviations.resize(n);
		mCNNConvolutionalLayers.resize(n);
		mCNNConvolutionalLayersDft.resize(n);
		mCNNSubsamplingLayers.resize(n);
		mCNNFullyConnectedLayers.resize(n);
		mCNNLayerTypes.resize(n);
		mCNNFullyConnectedLayersBias.resize(n);
		mCNNConvolutionalLayersBias.resize(n);

		///////////////////////////////////////
		// Read orientations
		for (int i = 0; i < n; i++)
		{
			cv::Mat_<double> orientation_tmp;
			ReadMatBin(stream, orientation_tmp);
		
			mOrientations[i] = cv::Vec3d(
				orientation_tmp.at<double>(0), 
				orientation_tmp.at<double>(1), 
				orientation_tmp.at<double>(2));

			// Convert from degrees to radians
			mOrientations[i] = mOrientations[i] * M_PI / 180.0;
		}

		///////////////////////////////////////
		// Read validators
		for (int i = 0; i < n; i++)
		{
			// Read in the mean images
			ReadMatBin(stream, mMeanImages[i]);
			ReadMatBin(stream, mStandardDeviations[i]);
			
			// Transpose them
			mMeanImages[i] = mMeanImages[i].t();
			mStandardDeviations[i] = mStandardDeviations[i].t();
			
			// Reading in CNNs
			int network_depth;
			stream.read ((char*)&network_depth, 4);
	
			mCNNLayerTypes[i].resize(network_depth);

			for(int layer = 0; layer < network_depth; ++layer)
			{
				int layer_type;
				stream.read ((char*)&layer_type, 4);
				mCNNLayerTypes[i][layer] = layer_type;

				// convolutional
				if(layer_type == 0)
				{
					int num_in_maps;
					int num_kernels;
					
					// Read the number of input maps and kernels for each input map
					stream.read ((char*)&num_in_maps, 4);
					stream.read ((char*)&num_kernels, 4);

					vector<vector<cv::Mat_<float>>> kernels;
					vector<vector<pair<int, cv::Mat_<double>>>> kernel_dfts;

					kernels.resize(num_in_maps);
					kernel_dfts.resize(num_in_maps);

					vector<float> biases;
					for (int k = 0; k < num_kernels; ++k)
					{
						float bias;
						stream.read ((char*)&bias, 4);
						biases.push_back(bias);
					}

					mCNNConvolutionalLayersBias[i].push_back(biases);

					// For every input map
					for (int in = 0; in < num_in_maps; ++in)
					{
						kernels[in].resize(num_kernels);
						kernel_dfts[in].resize(num_kernels);

						// For every kernel on that input map
						for (int k = 0; k < num_kernels; ++k)
						{
							ReadMatBin(stream, kernels[in][k]);
								
							// Flip the kernel in order to do convolution and not correlation
							cv::flip(kernels[in][k], kernels[in][k], -1);
						}
					}

					mCNNConvolutionalLayers[i].push_back(kernels);
					mCNNConvolutionalLayersDft[i].push_back(kernel_dfts);
				}
				else if(layer_type == 1)
				{
					// Subsampling layer
					int scale;
					stream.read ((char*)&scale, 4);

					mCNNSubsamplingLayers[i].push_back(scale);
				}
				else if(layer_type == 2)
				{
					float bias;
					stream.read ((char*)&bias, 4);

					mCNNFullyConnectedLayersBias[i].push_back(bias);

					// Fully connected layer
					cv::Mat_<float> weights;
					ReadMatBin(stream, weights);
					mCNNFullyConnectedLayers[i].push_back(weights);
				}
			}
					
			// Read in the piece-wise affine warps
			mPaws[i].Read(stream);
		}

		// close stream
		stream.close();
	}
	else
	{
		cout << "WARNING: Can't find the Face checker location" << endl;
	}
}

//===========================================================================

double DetectionValidator::Check(const cv::Vec3d& orientation, const cv::Mat_<uchar>& intensity_img, const cv::Mat_<double>& detected_landmarks)
{
	// SVR/NN not supported anymore
	assert(mValidatorType == Type::CNN);

	size_t VIEWIDX = GetViewId(orientation);
	
	// The warped (cropped) image, corresponding to a face lying withing the detected lanmarks
	cv::Mat_<double> warped_img;
	
	// the piece-wise affine image
	cv::Mat_<double> intensity_img_double;
	intensity_img.convertTo(intensity_img_double, CV_64F);

	mPaws[VIEWIDX].Warp(intensity_img_double, warped_img, detected_landmarks);

	////////////////////////////////

	cv::Mat_<double> feature_vec;
	NormaliseWarpedToVector(warped_img, feature_vec, VIEWIDX);

	// Create a normalised image from the crop vector
	cv::Mat_<float> img(warped_img.size(), 0.0);
	img = img.t();

	cv::Mat mask = mPaws[VIEWIDX].getPixelMask().t();

	cv::MatIterator_<uchar>  mask_it    = mask.begin<uchar>();
	cv::MatIterator_<double> feature_it = feature_vec.begin();
	cv::MatIterator_<float>  img_it     = img.begin();

	// if is within mask
	// assign the feature to image if it is within the mask
	for (int i = 0; i < img.cols; ++i)	
		for (int j = 0; j < img.rows; ++j, ++mask_it, ++img_it)			
			if (*mask_it)
				*img_it = (float)*feature_it++;
			
	img = img.t();

	int cnn_layer = 0;
	int subsample_layer = 0;
	int fully_connected_layer = 0;

	vector<cv::Mat_<float>> input_maps;
	input_maps.push_back(img);

	vector<cv::Mat_<float>> outputs;

	for (size_t layer = 0; layer < mCNNLayerTypes[VIEWIDX].size(); ++layer)
	{
		// Determine layer type
		const int layer_type = mCNNLayerTypes[VIEWIDX][layer];

		// Convolutional layer
		if (layer_type == 0)
		{
			vector<cv::Mat_<float>> outputs_kern;
			for (size_t in = 0; in < input_maps.size(); ++in)
			{
				cv::Mat_<float> input_image = input_maps[in];

				// Useful precomputed data placeholders for quick correlation (convolution)
				cv::Mat_<double> input_image_dft;
				cv::Mat integral_image;
				cv::Mat integral_image_sq;

				for (size_t k = 0; k < mCNNConvolutionalLayers[VIEWIDX][cnn_layer][in].size(); ++k)
				{
					cv::Mat_<float> kernel = mCNNConvolutionalLayers[VIEWIDX][cnn_layer][in][k];

					// The convolution (with precomputation)
					cv::Mat_<float> output;
					if (mCNNConvolutionalLayersDft[VIEWIDX][cnn_layer][in][k].second.empty())
					{
						std::map<int, cv::Mat_<double> > precomputed_dft;

						matchTemplate_m(input_image, input_image_dft, integral_image, integral_image_sq, kernel, precomputed_dft, output, CV_TM_CCORR);

						mCNNConvolutionalLayersDft[VIEWIDX][cnn_layer][in][k].first = precomputed_dft.begin()->first;
						mCNNConvolutionalLayersDft[VIEWIDX][cnn_layer][in][k].second = precomputed_dft.begin()->second;
					}
					else
					{
						std::map<int, cv::Mat_<double> > precomputed_dft;
						precomputed_dft[mCNNConvolutionalLayersDft[VIEWIDX][cnn_layer][in][k].first] = mCNNConvolutionalLayersDft[VIEWIDX][cnn_layer][in][k].second;
						matchTemplate_m(input_image, input_image_dft, integral_image, integral_image_sq, kernel, precomputed_dft, output, CV_TM_CCORR);
					}

					// Combining the maps
					if (in == 0)
					{
						outputs_kern.push_back(output);
					}
					else
					{
						outputs_kern[k] = outputs_kern[k] + output;
					}
				}
			}

			outputs.clear();
			for (size_t k = 0; k < mCNNConvolutionalLayers[VIEWIDX][cnn_layer][0].size(); ++k)
			{
				// Apply the sigmoid
				cv::exp(-outputs_kern[k] - mCNNConvolutionalLayersBias[VIEWIDX][cnn_layer][k], outputs_kern[k]);
				outputs_kern[k] = 1.0 / (1.0 + outputs_kern[k]);

				outputs.push_back(outputs_kern[k]);
			}

			cnn_layer++;
		}

		else if (layer_type == 1)
		{
			// Subsampling layer
			int scale = mCNNSubsamplingLayers[VIEWIDX][subsample_layer];

			cv::Mat kx = cv::Mat::ones(2, 1, CV_32F)*1.0f / scale;
			cv::Mat ky = cv::Mat::ones(1, 2, CV_32F)*1.0f / scale;

			vector<cv::Mat_<float>> outputs_sub;
			for (size_t in = 0; in < input_maps.size(); ++in)
			{

				cv::Mat_<float> conv_out;

				cv::sepFilter2D(input_maps[in], conv_out, CV_32F, kx, ky);
				conv_out = conv_out(cv::Rect(1, 1, conv_out.cols - 1, conv_out.rows - 1));

				int res_rows = conv_out.rows / scale;
				int res_cols = conv_out.cols / scale;

				if (conv_out.rows % scale != 0)
				{
					res_rows++;
				}
				if (conv_out.cols % scale != 0)
				{
					res_cols++;
				}

				cv::Mat_<float> sub_out(res_rows, res_cols);
				for (int w = 0; w < conv_out.cols; w += scale)
				{
					for (int h = 0; h < conv_out.rows; h += scale)
					{
						sub_out.at<float>(h / scale, w / scale) = conv_out(h, w);
					}
				}
				outputs_sub.push_back(sub_out);
			}
			outputs = outputs_sub;
			subsample_layer++;
		}

		else if (layer_type == 2)
		{
			// Concatenate all the maps
			cv::Mat_<float> input_concat = input_maps[0].t();
			input_concat = input_concat.reshape(0, 1);

			for (size_t in = 1; in < input_maps.size(); ++in)
			{
				cv::Mat_<float> add = input_maps[in].t();
				add = add.reshape(0, 1);
				cv::hconcat(input_concat, add, input_concat);
			}

			input_concat = input_concat * mCNNFullyConnectedLayers[VIEWIDX][fully_connected_layer].t();

			cv::exp(-input_concat - mCNNFullyConnectedLayersBias[VIEWIDX][fully_connected_layer], input_concat);
			input_concat = 1.0 / (1.0 + input_concat);

			outputs.clear();
			outputs.push_back(input_concat);

			fully_connected_layer++;
		}

		// Set the outputs of this layer to inputs of the next
		input_maps = outputs;
	}

	// Turn it to -1, 1 range
	double dec = (outputs[0].at<float>(0) - 0.5) * 2.0;

	return dec;
}

void DetectionValidator::NormaliseWarpedToVector(const cv::Mat_<double>& warped_img, cv::Mat_<double>& feature_vec, int view_id)
{
	cv::Mat_<double> warped_t = warped_img.t();
	
	// the vector to be filled with paw values
	cv::Mat_<double> vec(mPaws[view_id].getNumberOfPixels(), 1);
	
	// the mask indicating if point is within or outside the face region	
	cv::Mat maskT = mPaws[view_id].getPixelMask().t();

	cv::MatIterator_<double> vp = vec.begin();
	cv::MatIterator_<double> cp = warped_t.begin();
	cv::MatIterator_<uchar>  mp = maskT.begin<uchar>();

	// if is within mask
	for (int i = 0; i < warped_img.cols; ++i)	
		for (int j = 0; j < warped_img.rows; ++j, ++mp, ++cp)					
			if(*mp)
				*vp++ = *cp;
			
	// Local normalisation
	cv::Scalar mean;
	cv::Scalar std;
	cv::meanStdDev(vec, mean, std);

	// subtract the mean image
	vec -= mean[0];

	// Normalise the image
	if (std[0] == 0.0)	
		std[0] = 1.0;
		
	vec /= std[0];

	// Global normalisation
	feature_vec = (vec - mMeanImages[view_id])  / mStandardDeviations[view_id];
}

// Getting the closest view center based on orientation
size_t DetectionValidator::GetViewId(const cv::Vec3d& orientation) const
{
	size_t id = 0;
	double dbest = -1.0;

	const size_t COUNT = mOrientations.size();
	for(size_t i = 0; i < COUNT; i++)
	{	
		// Distance to current view
		const double D = cv::norm(orientation, mOrientations[i]);

		// better than last
		if(i == 0 || D < dbest)
		{
			dbest = D;
			id = i;
		}
	}

	return id;
}
