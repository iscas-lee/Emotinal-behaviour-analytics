

#ifndef __ACTION_UNIT_h_
#define __ACTION_UNIT_h_

#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>

namespace OpenFace
{

/// <summary>
/// Action Unit Enumeration
/// </summary>
enum ActionUnit
{
	AU01 = 1,  AU02 = 2,  AU03 = 3,  AU04 = 4,  AU05 = 5,
	AU06 = 6,  AU07 = 7,  AU08 = 8,  AU09 = 9,  AU10 = 10,
	AU11 = 11, AU12 = 12, AU13 = 13, AU14 = 14, AU15 = 15,
	AU16 = 16, AU17 = 17, AU18 = 18, AU19 = 19, AU20 = 20,
	AU21 = 21, AU22 = 22, AU23 = 23, AU24 = 24, AU25 = 25,
	AU26 = 26, AU27 = 27, AU28 = 28, AU29 = 29, AU30 = 30,
	AU31 = 31, AU32 = 32, AU33 = 33, AU34 = 34, AU35 = 35,
	AU36 = 36, AU37 = 37, AU38 = 38, AU39 = 39, AU40 = 40,
	AU41 = 41, AU42 = 42, AU43 = 43, AU44 = 44, AU45 = 45
};

/// <summary>
/// Set of Action Unit Values
/// </summary>
struct ActionUnitValues
{
	double AU01;
	double AU02;
	double AU04;
	double AU05;
	double AU06;
	double AU07;
	double AU09;
	double AU10;
	double AU12;
	double AU14;
	double AU15;
	double AU17;
	double AU20;
	double AU23;
	double AU25;
	double AU26;
	double AU45;
};

//=========================================================================================================

/// <summary>
/// Collection of linear SVR regressors for AU prediction
/// </summary>
class SVMDynamicLinear
{
public:
	SVMDynamicLinear() { }

	// Predict the AU from HOG appearance of the face
	void Predict(
		ActionUnitValues&       predictions,
		const cv::Mat_<double>& fhog_descriptor,
		const cv::Mat_<double>& geom_params,
		const cv::Mat_<double>& running_median,
		const cv::Mat_<double>& running_median_geom);

	// Reading in the model (or adding to it)
	void Read(std::istream& stream, const ActionUnit& au_name);

protected:
	std::vector<ActionUnit>  mAUNames;        // The names of Action Units this model is responsible for
	cv::Mat_<double>         mMeans;          // For normalisation
	cv::Mat_<double>         mSupportVectors; // For actual prediction
	cv::Mat_<double>         mBiases;         // For actual prediction
	std::vector<double>      mPosClasses;
	std::vector<double>      mNegClasses;
};

//=========================================================================================================

/// <summary>
/// Collection of linear SVR regressors for AU prediction
/// </summary>
class SVMStaticLinear
{
public:
	SVMStaticLinear() { }

	// Predict the AU from HOG appearance of the face
	void Predict(
		ActionUnitValues&       predictions,
		const cv::Mat_<double>& fhog_descriptor,
		const cv::Mat_<double>& geom_params);

	// Reading in the model (or adding to it)
	void Read(std::istream& stream, const ActionUnit& au_name);

protected:
	std::vector<ActionUnit>  mAUNames;        // The names of Action Units this model is responsible for
	cv::Mat_<double>         mMeans;          // For normalisation
	cv::Mat_<double>         mSupportVectors; // For actual prediction
	cv::Mat_<double>         mBiases;         // For actual prediction
	std::vector<double>      mPosClasses;
	std::vector<double>      mNegClasses;
};

//=========================================================================================================

/// <summary>
/// Collection of linear SVR regressors for AU prediction that 
/// uses per person face nomalisation with the help of a running median
/// </summary>
class SVRDynamicLinear
{
public:
	SVRDynamicLinear() { }

	// Predict the AU from HOG appearance of the face
	void Predict(
		ActionUnitValues&       predictions,
		const cv::Mat_<double>& descriptor,
		const cv::Mat_<double>& geom_params,
		const cv::Mat_<double>& running_median,
		const cv::Mat_<double>& running_median_geom);

	// Reading in the model (or adding to it)
	void Read(std::istream& stream, const ActionUnit& au_name);

protected:
	std::vector<ActionUnit>  mAUNames;        // The names of Action Units this model is responsible for
	cv::Mat_<double>         mMeans;          // For normalisation
	cv::Mat_<double>         mSupportVectors; // For actual prediction
	cv::Mat_<double>         mBiases;         // For actual prediction
	std::vector<double>      mCutOffs;        // For AU callibration (see the OpenFace paper)
};

//=========================================================================================================

/// <summary>
/// Collection of linear SVR regressors for AU prediction
/// </summary>
class SVRStaticLinear
{
public:
	SVRStaticLinear() { }

	// Predict the AU from HOG appearance of the face
	void Predict(
		ActionUnitValues&       predictions,
		const cv::Mat_<double>& fhog_descriptor,
		const cv::Mat_<double>& geom_params);

	// Reading in the model (or adding to it)
	void Read(std::istream& stream, const ActionUnit& au_name);

protected:
	std::vector<ActionUnit> mAUNames;        // The names of Action Units this model is responsible for
	cv::Mat_<double>        mMeans;          // For normalisation
	cv::Mat_<double>        mSupportVectors; // For actual prediction
	cv::Mat_<double>        mBiases;         // For actual prediction
};
}
#endif
