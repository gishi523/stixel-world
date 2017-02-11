#include <iostream>
#include <opencv2/opencv.hpp>
#include "stixel_world.h"

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml" << std::endl;
		return -1;
	}

	// stereo sgbm
	const int wsize = 11;
	const int numDisparities = 64;
	const int P1 = 8 * wsize * wsize;
	const int P2 = 32 * wsize * wsize;
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(0, numDisparities, wsize, P1, P2,
		0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);

	// input camera parameters
	const cv::FileStorage cvfs(argv[3], CV_STORAGE_READ);
	CV_Assert(cvfs.isOpened());
	const cv::FileNode node(cvfs.fs, NULL);
	const float focalLengthX = node["FocalLengthX"];
	const float focalLengthY = node["FocalLengthY"];
	const float principalPointX = node["CenterX"];
	const float principalPointY = node["CenterY"];
	const float baseline = node["BaseLine"];
	const float cameraHeight = node["Height"];
	const float cameraTilt = node["Tilt"];

	StixelWorld sw(focalLengthX, focalLengthY, principalPointX, principalPointY, 
		baseline, cameraHeight, cameraTilt);

	for (int frameno = 1;; frameno++)
	{
		char bufl[256], bufr[256];
		sprintf(bufl, argv[1], frameno);
		sprintf(bufr, argv[2], frameno);

		cv::Mat left = cv::imread(bufl, -1);
		cv::Mat right = cv::imread(bufr, -1);

		if (left.empty() || right.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		CV_Assert(left.size() == right.size() && left.type() == right.type());

		switch (left.type())
		{
		case CV_8U:
			// nothing to do
			break;
		case CV_16U:
			// conver to CV_8U
			double maxVal;
			cv::minMaxLoc(left, NULL, &maxVal);
			left.convertTo(left, CV_8U, 255 / maxVal);
			right.convertTo(right, CV_8U, 255 / maxVal);
			break;
		default:
			std::cerr << "unsupported image type." << std::endl;
			return -1;
		}

		// calculate dispaliry
		cv::Mat disp;
		ssgbm->compute(left, right, disp);
		disp.convertTo(disp, CV_32F, 1.0 / 16);

		// calculate stixels
		std::vector<Stixel> stixels;
		sw.compute(disp, stixels, 7);

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(left, draw, cv::COLOR_GRAY2BGRA);

		cv::Mat stixelImg = cv::Mat::zeros(left.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp, 64));

		draw = draw + 0.5 * stixelImg;

		cv::imshow("disparity", disp / 64);
		cv::imshow("stixels", draw);
		
		const char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
	}
}
