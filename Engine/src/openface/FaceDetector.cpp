#include "stdafx.h"

using namespace OpenFace;

FaceDetectorDlib::FaceDetectorDlib() : mFaceDetector(dlib::get_frontal_face_detector())
{
}

bool FaceDetectorDlib::DetectFaces(
	vector<cv::Rect_<double>>& o_regions, 
	const cv::Mat_<uchar>&     intensity, 
	std::vector<double>&       o_confidences)
{
	// CLNF expects the bounding box to encompass from eyebrow to chin in y, 
	// and from cheeck outline to cheeck outline in x, so we need to compensate
	// The scalings were learned using the Face Detections on LFPW and Helen 
	// using ground truth and detections from the HOG detector
	const double SCALING = 1.3;

	// upsample the input by scaling
	cv::Mat_<uchar> upsampled_intensity;
	cv::resize(intensity, upsampled_intensity, cv::Size((int)(intensity.cols * SCALING), (int)(intensity.rows * SCALING)));

	/////////////////////////////////////////////////////////////////////
	
	// get dlib image for cv mat
	dlib::cv_image<uchar> cv_grayscale(upsampled_intensity);

	// stores the detected faces
	std::vector<dlib::full_detection> face_detections;

	// run the actual detection
	mFaceDetector(cv_grayscale, face_detections, -0.2);

	/////////////////////////////////////////////////////////////////////
	
	// how many faces were detected
	const size_t NUM_DETECTIONS = face_detections.size();

	// Convert from int bounding box do a double one with corrections
	o_regions.resize(NUM_DETECTIONS);
	o_confidences.resize(NUM_DETECTIONS);

	for (size_t face = 0; face < NUM_DETECTIONS; ++face)
	{
		// detected face rectangle
		const dlib::rectangle& FACE_RECT = face_detections[face].rect.get_rect();

		// Move the face slightly to the right (as the width was made smaller) and
		// shift face down as OpenCV Haar Cascade detects the forehead as well, and we're not interested
		o_regions[face].x = (FACE_RECT.tl_corner().x() + 0.0389 * FACE_RECT.width()) / SCALING;
		o_regions[face].y = (FACE_RECT.tl_corner().y() + 0.1278 * FACE_RECT.height()) / SCALING;

		// Correct for scale
		o_regions[face].width = (face_detections[face].rect.get_rect().width() * 0.9611) / SCALING;
		o_regions[face].height = (face_detections[face].rect.get_rect().height() * 0.9388) / SCALING;

		o_confidences[face] = face_detections[face].detection_confidence;
	}

	return o_regions.size() > 0;
}

bool FaceDetectorDlib::DetectSingleFace(
	cv::Rect_<double>&     o_region, 
	const cv::Mat_<uchar>& intensity_img, 
	double&                confidence)
{
	// The tracker can return multiple faces
	vector<cv::Rect_<double>> face_detections;
	vector<double> confidences;

	// Find all faces
	const bool SUCCESS = DetectFaces(face_detections, intensity_img, confidences);

	// Found at least one face
	if (SUCCESS)
	{
		// keep the most confident one or 
		// the one closest to preference point if set
		double best_so_far = confidences[0];
		int bestIndex = 0;

		for (size_t i = 1; i < face_detections.size(); ++i)
		{
			const double dist = confidences[i];
			const bool better = dist > best_so_far;
			
			// Pick a closest face
			if (better)
			{
				best_so_far = dist;
				bestIndex = (int)i;
			}
		}

		o_region = face_detections[bestIndex];
		confidence = confidences[bestIndex];
	}

	// No face
	else
	{
		o_region = cv::Rect_<double>(0, 0, 0, 0);
		
		// A completely unreliable detection
		// (shouldn't really matter what is returned here)
		confidence = -2.0;
	}

	return SUCCESS;
}
