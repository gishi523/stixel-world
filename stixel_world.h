#ifndef __STIXEL_WORLD_H__
#define __STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>

struct Stixel
{
	int u;
	int vT;
	int vB;
	int width;
	float disp;
};

class StixelWorld
{
public:

	enum
	{
		ROAD_ESTIMATION_AUTO = 0,
		ROAD_ESTIMATION_CAMERA
	};

	struct CameraParameters
	{
		float fu;
		float fv;
		float u0;
		float v0;
		float baseline;
		float height;
		float tilt;

		// default settings
		CameraParameters()
		{
			fu = 1.f;
			fv = 1.f;
			u0 = 0.f;
			v0 = 0.f;
			baseline = 0.2f;
			height = 1.f;
			tilt = 0.f;
		}
	};

	struct Parameters
	{
		// stixel width
		int stixelWidth;

		// minimum and maximum disparity
		int minDisparity;
		int maxDisparity;

		// road disparity estimation
		int roadEstimation;

		// camera parameters
		CameraParameters camera;

		// default settings
		Parameters()
		{
			// stixel width
			stixelWidth = 7;

			// minimum and maximum disparity
			minDisparity = -1;
			maxDisparity = 64;

			// road disparity estimation
			roadEstimation = ROAD_ESTIMATION_AUTO;

			// camera parameters
			camera = CameraParameters();
		}
	};

	StixelWorld(const Parameters& param = Parameters());
	void compute(const cv::Mat& disp, std::vector<Stixel>& stixels);

private:
	std::vector<int> lowerPath_, upperPath_;
	Parameters param_;
};

#endif // !__STIXEL_WORLD_H__