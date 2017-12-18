
#ifndef __FACE_DETECTOR__h_
#define __FACE_DETECTOR__h_

#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

namespace OpenFace
{

/// <summary>
/// Face detector from Dlib
/// </summary>
class FaceDetectorDlib
{
public:
	FaceDetectorDlib();

	// Face detection using HOG-SVM classifier
	bool DetectFaces(std::vector<cv::Rect_<double>>& o_regions, const cv::Mat_<uchar>& intensity, std::vector<double>& confidences);

	// The preference point allows for disambiguation if multiple faces are present (pick the closest one), if it is not set the biggest face is chosen
	bool DetectSingleFace(cv::Rect_<double>& o_region, const cv::Mat_<uchar>& intensity, double& confidence);

protected:
	dlib::frontal_face_detector mFaceDetector;
};
}
#endif
