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

#ifndef __FACE_UTIL_h_
#define __FACE_UTIL_h_

#include <fstream>
#include <opencv2/core/core.hpp>

using namespace std;

namespace OpenFace
{
	//===========================================================================
	#pragma region ANGLE CONVERSION HELPERS
	//===========================================================================

	/// <summary>
	/// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	/// </summary>
	__forceinline cv::Matx33d Euler2RotationMatrix(const cv::Vec3d& eulerAngles)
	{
		cv::Matx33d rotation_matrix;

		const double s1 = sin(eulerAngles[0]);
		const double s2 = sin(eulerAngles[1]);
		const double s3 = sin(eulerAngles[2]);

		const double c1 = cos(eulerAngles[0]);
		const double c2 = cos(eulerAngles[1]);
		const double c3 = cos(eulerAngles[2]);

		rotation_matrix(0, 0) = c2 * c3;
		rotation_matrix(0, 1) = -c2 *s3;
		rotation_matrix(0, 2) = s2;
		rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
		rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
		rotation_matrix(1, 2) = -c2 * s1;
		rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
		rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
		rotation_matrix(2, 2) = c1 * c2;

		return rotation_matrix;
	}

	/// <summary>
	/// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	/// </summary>
	__forceinline cv::Vec3d RotationMatrix2Euler(const cv::Matx33d& rotation_matrix)
	{
		const double q0 = sqrt(1 + rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2)) / 2.0;
		const double q1 = (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / (4.0*q0);
		const double q2 = (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / (4.0*q0);
		const double q3 = (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / (4.0*q0);

		const double t1 = 2.0 * (q0*q2 + q1*q3);

		const double yaw = asin(2.0 * (q0*q2 + q1*q3));
		const double pitch = atan2(2.0 * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3);
		const double roll = atan2(2.0 * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3);

		return cv::Vec3d(pitch, yaw, roll);
	}

	__forceinline cv::Vec3d Euler2AxisAngle(const cv::Vec3d& euler)
	{
		cv::Matx33d rotMatrix = Euler2RotationMatrix(euler);
		cv::Vec3d axis_angle;
		cv::Rodrigues(rotMatrix, axis_angle);
		return axis_angle;
	}

	__forceinline cv::Vec3d AxisAngle2Euler(const cv::Vec3d& axis_angle)
	{
		cv::Matx33d rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return RotationMatrix2Euler(rotation_matrix);
	}

	__forceinline cv::Matx33d AxisAngle2RotationMatrix(const cv::Vec3d& axis_angle)
	{
		cv::Matx33d rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return rotation_matrix;
	}

	__forceinline cv::Vec3d RotationMatrix2AxisAngle(const cv::Matx33d& rotation_matrix)
	{
		cv::Vec3d axis_angle;
		cv::Rodrigues(rotation_matrix, axis_angle);
		return axis_angle;
	}
	#pragma endregion
	//===========================================================================

	//============================================================================
	#pragma region MISC
	//============================================================================

	/// <summary>
	/// Orthonormalising the 3x3 rotation matrix
	/// </summary>
	__forceinline void Orthonormalise(cv::Matx33d &R)
	{
		cv::SVD svd(R, cv::SVD::MODIFY_A);

		// get the orthogonal matrix from the initial rotation matrix
		cv::Mat_<double> X = svd.u*svd.vt;

		// This makes sure that the handedness is preserved and no reflection happened
		// by making sure the determinant is 1 and not -1
		cv::Mat_<double> W = cv::Mat_<double>::eye(3, 3);

		W(2, 2) = determinant(X);

		cv::Mat Rt = svd.u * W * svd.vt;

		Rt.copyTo(R);
	}

	// Is the point (x0,y0) on same side as a half-plane defined by (x1,y1), (x2, y2), and (x3, y3)
	__forceinline bool sameSide(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3)
	{
		const double x = (x3 - x2)*(y0 - y2) - (x0 - x2)*(y3 - y2);
		const double y = (x3 - x2)*(y1 - y2) - (x1 - x2)*(y3 - y2);

		return x * y >= 0;
	}

	// if point (x0, y0) is on same side for all three half-planes it is in a triangle
	__forceinline bool pointInTriangle(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3)
	{
		const bool same_1 = sameSide(x0, y0, x1, y1, x2, y2, x3, y3);
		const bool same_2 = sameSide(x0, y0, x2, y2, x1, y1, x3, y3);
		const bool same_3 = sameSide(x0, y0, x3, y3, x1, y1, x2, y2);

		return same_1 && same_2 && same_3;
	}

	__forceinline cv::Point3f RaySphereIntersect(cv::Point3f rayOrigin, cv::Point3f rayDir, cv::Point3f sphereOrigin, float sphereRadius)
	{
		float dx = rayDir.x;
		float dy = rayDir.y;
		float dz = rayDir.z;
		float x0 = rayOrigin.x;
		float y0 = rayOrigin.y;
		float z0 = rayOrigin.z;
		float cx = sphereOrigin.x;
		float cy = sphereOrigin.y;
		float cz = sphereOrigin.z;
		float r = sphereRadius;

		float a = dx*dx + dy*dy + dz*dz;
		float b = 2 * dx*(x0 - cx) + 2 * dy*(y0 - cy) + 2 * dz*(z0 - cz);
		float c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2 * (cx*x0 + cy*y0 + cz*z0) - r*r;

		float disc = b*b - 4 * a*c;

		float t = (-b - sqrt(b*b - 4 * a*c)) / 2 * a;

		// This implies that the lines did not intersect, point straight ahead
		if (b*b - 4 * a*c < 0)
			return cv::Point3f(0, 0, -1);

		return rayOrigin + rayDir * t;
	}

	__forceinline void Project(cv::Mat_<double>& dest, const cv::Mat_<double>& src, double fx, double fy, double cx, double cy)
	{
		const int NUM_POINTS = src.rows;

		// create mat
		dest = cv::Mat_<double>(NUM_POINTS, 2, 0.0);

		cv::Mat_<double>::const_iterator mData = src.begin();
		cv::Mat_<double>::iterator projected = dest.begin();

		for (int i = 0; i < NUM_POINTS; i++)
		{
			// Get the points
			const double X = *(mData++);
			const double Y = *(mData++);
			const double Z = *(mData++);

			double x;
			double y;

			// if depth is 0 the projection is different
			if (Z != 0)
			{
				x = ((X * fx / Z) + cx);
				y = ((Y * fy / Z) + cy);
			}
			else
			{
				x = X;
				y = Y;
			}

			// Project and store in dest matrix
			(*projected++) = x;
			(*projected++) = y;
		}
	}
	#pragma endregion
	//============================================================================

	//============================================================================
	#pragma region KABSCH ALGORITHM
	//============================================================================

	// Using Kabsch's algorithm for aligning shapes
	// This assumes that align_from and align_to are already mean normalised
	__forceinline cv::Matx22d AlignShapesKabsch2D(const cv::Mat_<double>& align_from, const cv::Mat_<double>& align_to)
	{
		cv::SVD svd(align_from.t() * align_to);

		// make sure no reflection is there
		// corr ensures that we do only rotaitons and not reflections
		const double D = cv::determinant(svd.vt.t() * svd.u.t());

		cv::Matx22d corr = cv::Matx22d::eye();
		corr(1, 1) = (D > 0.0) ? 1 : -1;

		cv::Matx22d R;
		cv::Mat(svd.vt.t() * cv::Mat(corr) * svd.u.t()).copyTo(R);

		return R;
	}

	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	__forceinline cv::Matx22d AlignShapesWithScale(cv::Mat_<double>& src, cv::Mat_<double> dst)
	{
		const int n = src.rows;

		// First we mean normalise both src and dst
		double mean_src_x = cv::mean(src.col(0))[0];
		double mean_src_y = cv::mean(src.col(1))[0];
		double mean_dst_x = cv::mean(dst.col(0))[0];
		double mean_dst_y = cv::mean(dst.col(1))[0];

		cv::Mat_<double> src_mean_normed = src.clone();
		src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
		src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

		cv::Mat_<double> dst_mean_normed = dst.clone();
		dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
		dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

		// Find the scaling factor of each
		cv::Mat src_sq;
		cv::Mat dst_sq;

		cv::pow(src_mean_normed, 2, src_sq);
		cv::pow(dst_mean_normed, 2, dst_sq);

		const double s_src = sqrt(cv::sum(src_sq)[0] / n);
		const double s_dst = sqrt(cv::sum(dst_sq)[0] / n);

		src_mean_normed = src_mean_normed / s_src;
		dst_mean_normed = dst_mean_normed / s_dst;

		const double s = s_dst / s_src;

		// Get the rotation
		cv::Matx22d R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);

		cv::Matx22d	A;
		cv::Mat(s * R).copyTo(A);

		cv::Mat_<double> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
		cv::Mat_<double> offset = dst - aligned;

		double t_x = cv::mean(offset.col(0))[0];
		double t_y = cv::mean(offset.col(1))[0];

		return A;
	}
	#pragma endregion
	//============================================================================

	//============================================================================
	#pragma region MATRIX AND OTHER PARSERS
	//============================================================================

	/// <summary>
	/// Read a matrix from a text-stream
	/// </summary>
	__forceinline void ReadMat(std::istream& stream, cv::Mat& output_mat)
	{
		int row, col, type;

		// read in the number of rows, columns and the data type
		stream >> row >> col >> type;

		// create mat to store data
		output_mat = cv::Mat(row, col, type);

		switch (output_mat.type())
		{
		case CV_64FC1:
		{
			cv::MatIterator_<double> begin_it = output_mat.begin<double>();
			cv::MatIterator_<double> end_it = output_mat.end<double>();

			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32FC1:
		{
			cv::MatIterator_<float> begin_it = output_mat.begin<float>();
			cv::MatIterator_<float> end_it = output_mat.end<float>();

			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_32SC1:
		{
			cv::MatIterator_<int> begin_it = output_mat.begin<int>();
			cv::MatIterator_<int> end_it = output_mat.end<int>();
			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		case CV_8UC1:
		{
			cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
			cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
			while (begin_it != end_it)
			{
				stream >> *begin_it++;
			}
		}
		break;
		default:
			printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__, __LINE__, output_mat.type()); abort();
		}
	}

	/// <summary>
	/// Read a matrix from a binary-stream
	/// </summary>
	__forceinline void ReadMatBin(std::istream& stream, cv::Mat& output_mat)
	{
		int row, col, type;

		// read in the number of rows, columns and the data type
		stream.read((char*)&row, 4);
		stream.read((char*)&col, 4);
		stream.read((char*)&type, 4);

		// create Mat
		output_mat = cv::Mat(row, col, type);

		// calculate the data size of the mat
		const int SIZE = output_mat.rows * output_mat.cols * (int)output_mat.elemSize();

		// read all mat data
		stream.read((char*)output_mat.data, SIZE);
	}

	/// <summary>
	/// Skipping lines that start with # (together with empty lines)
	/// </summary>
	__forceinline void SkipComments(std::istream& stream)
	{
		while (stream.peek() == '#' || stream.peek() == '\n' || stream.peek() == ' ' || stream.peek() == '\r')
		{
			std::string skipped;
			std::getline(stream, skipped);
		}
	}
	#pragma endregion
	//============================================================================

	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================
	// This is a modified version of openCV code that allows for precomputed dfts of templates and for precomputed dfts of an image
	// _img is the input img, _img_dft it's dft (optional), _integral_img the images integral image (optional), squared integral image (optional), 
	// templ is the template we are convolving with, templ_dfts it's dfts at varying windows sizes (optional),  _result - the output, method the type of convolution
	void matchTemplate_m(
		const cv::Mat_<float>& input_img, 
		cv::Mat_<double>& img_dft, 
		cv::Mat& _integral_img, 
		cv::Mat& _integral_img_sq, 
		const cv::Mat_<float>&  templ, 
		map<int, cv::Mat_<double>>& templ_dfts, 
		cv::Mat_<float>& result, int method);
}
#endif
