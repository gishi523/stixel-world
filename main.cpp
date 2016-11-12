#include <iostream>
#include <opencv2/opencv.hpp>
#include "stixel_world.h"
#include "timer.h"

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

static cv::Scalar dispToColor(double disp, double maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	cv::Point tl(stixel.u - radius, stixel.vT);
	cv::Point br(stixel.u + radius, stixel.vB);
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
	int wsize = 11;
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(0, 64, wsize, 8 * wsize * wsize, 32 * wsize * wsize,
		0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);

	// input camera parameters
	cv::FileStorage cvfs(argv[3], CV_STORAGE_READ);
	CV_Assert(cvfs.isOpened());
	cv::FileNode node(cvfs.fs, NULL);
	float focalLengthX = node["FocalLengthX"];
	float focalLengthY = node["FocalLengthY"];
	float principalPointX = node["CenterX"];
	float principalPointY = node["CenterY"];
	float baseline = node["BaseLine"];
	float cameraHeight = node["Height"];
	float cameraTilt = node["Tilt"];

	StixelWrold sw(focalLengthX, focalLengthY, principalPointX, principalPointY, 
		baseline, cameraHeight, cameraTilt);

	Timer t;

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
			double maxval;
			cv::minMaxLoc(left, NULL, &maxval);
			left.convertTo(left, CV_8U, 255 / maxval);
			right.convertTo(right, CV_8U, 255 / maxval);
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

		t.start();
		t.stop();
		std::cout << "computation time: " << t.getms() << "[msec]" << std::endl;

		// draw free space
		cv::Mat draw;
		cv::cvtColor(left, draw, cv::COLOR_GRAY2BGRA);

		cv::Mat stixelimg = cv::Mat::zeros(left.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelimg, stixel, dispToColor(stixel.disp, 64));

		draw = draw + 0.5 * stixelimg;
#else
		for (int u = 0; u < left.cols; u++)
		{
			cv::circle(draw, cv::Point(u, sw.lowerPath[u]), 1, cv::Scalar(0, 255, 255));
			cv::circle(draw, cv::Point(u, sw.upperPath[u]), 1, cv::Scalar(0, 0, 255));
		}
#endif

		cv::imshow("disparity", disp / 64);
		cv::imshow("draw", draw);
		
		char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
	}
}