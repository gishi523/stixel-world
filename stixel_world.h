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

class StixelWrold
{
public:
	StixelWrold() = delete;

	StixelWrold(
		float focalLengthX,
		float focalLengthY,
		float principalPointX,
		float principalPointY,
		float baseline,
		float cameraHeight,
		float cameraTilt);

	void compute(const cv::Mat& disp, std::vector<Stixel>& stixels, int stixelWidth = 7);

private:
	float focalLengthX_;
	float focalLengthY_;
	float principalPointX_;
	float principalPointY_;
	float baseline_;
	float cameraHeight_;
	float cameraTilt_;
};

#endif // !__STIXEL_WORLD_H__