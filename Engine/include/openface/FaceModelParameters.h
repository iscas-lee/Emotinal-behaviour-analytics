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

#ifndef __FACE_MODEL_PARAM_H
#define __FACE_MODEL_PARAM_H

#include <vector>

using namespace std;

namespace OpenFace
{

struct FaceModelParameters
{
	int  num_optimisation_iteration;  // A number of RLMS or NU-RLMS iterations
	bool limit_pose;                  // Should pose be limited to 180 degrees frontal
	bool validate_detections;         // Should face validation be done
	double validation_boundary;       // Landmark detection validator boundary for correct detection, the regressor output -1 (perfect alignment) 1 (bad alignment)
	vector<int> window_sizes_small;   // Used when tracking is going well
	vector<int> window_sizes_init;    // Used when initialising or tracking fails
	vector<int> window_sizes_current; // Used for the current frame
	double face_template_scale;	      // How big is the tracking template that helps with large motions
	bool use_face_template;
	double sigma;                     // this is used for the smooting of response maps (KDE sigma)
	double reg_factor;	              // weight put to regularisation
	double weight_factor;             // factor for weighted least squares
	int reinit_video_every;           // How often should face detection be used to attempt reinitialisation, every n frames (set to negative not to reinit)
	bool refine_parameters;           // Should the parameters be refined for different scales
	bool track_gaze;                  // Using the brand new and experimental gaze tracker

	FaceModelParameters();
};

}

#endif // __LANDMARK_DETECTOR_PARAM_H
