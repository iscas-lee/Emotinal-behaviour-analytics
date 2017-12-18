#pragma once

#include <opencv2/core/core.hpp>

namespace OpenFace
{
struct GlobalParameters
{
public:
	double    scale;
	cv::Vec3d orient;
	double tx;
	double ty;

	GlobalParameters() :
		scale(1.0),
		orient(0.0, 0.0, 0.0),
		tx(0.0),
		ty(0.0)
	{ }

	GlobalParameters(const GlobalParameters& other) :
		scale(other.scale), 
		orient(other.orient),
		tx(other.tx), 
		ty(other.ty)
	{ }

	GlobalParameters(double scale, cv::Vec3d orient, double tx, double ty)
	{
		this->scale = scale;
		this->orient = orient;
		this->tx = tx;
		this->ty = ty;
	}

	void reset()
	{
		this->scale = 1.0;
		this->orient[0] = 0.0;
		this->orient[1] = 0.0;
		this->orient[2] = 0.0;
		this->tx = 0.0;
		this->ty = 0.0;
	}
};
}