// Precompiled headers stuff

#ifndef __STDAFX_h_
#define __STDAFX_h_

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

// IplImage stuff
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>

// C++ stuff
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <string>

// TBB includes
#include <tbb/tbb.h>

// dlib (For FHOG visualisation)
#include <dlib/opencv.h>

// libsvm
#include <libsvm/svm.h>

// Math includes
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Local includes
#include <FaceQueue.h>
#include <FaceWorker.h>
#include <FaceActionUnit.h>
#include <FaceResult.h>
#include <FaceGlobalParameters.h>
#include <FaceUtil.h>
#include <FaceDetector.h>
#include <FaceValidator.h>
#include <FaceModel.h>
#include <FaceModelParameters.h>
#include <FaceAnalyser.h>
#include <FaceCCNFPatchExpert.h>
#include <FacePatchExperts.h>
#include <FaceSystem.h>
#include <PAW.h>
#include <PDM.h>

#endif
