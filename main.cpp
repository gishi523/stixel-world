#include <opencv2/opencv.hpp>
#include <chrono>
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
	StixelWorld::Parameters param;
	param.camera.fu = node["FocalLengthX"];
	param.camera.fv = node["FocalLengthY"];
	param.camera.u0 = node["CenterX"];
	param.camera.v0 = node["CenterY"];
	param.camera.baseline = node["BaseLine"];
	param.camera.height = node["Height"];
	param.camera.tilt = node["Tilt"];
	param.minDisparity = -1;
	param.maxDisparity = numDisparities;

	StixelWorld stixelWorld(param);

	for (int frameno = 1;; frameno++)
	{
		char buf1[256];
		char buf2[256];
		sprintf(buf1, argv[1], frameno);
		sprintf(buf2, argv[2], frameno);

		cv::Mat I1 = cv::imread(buf1, -1);
		cv::Mat I2 = cv::imread(buf2, -1);

		if (I1.empty() || I2.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());

		switch (I1.type())
		{
		case CV_8U:
			// nothing to do
			break;
		case CV_16U:
			// conver to CV_8U
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
			I2.convertTo(I2, CV_8U);
			break;
		default:
			std::cerr << "unsupported image type." << std::endl;
			return -1;
		}

		// calculate dispaliry
		cv::Mat disparity;
		ssgbm->compute(I1, I2, disparity);
		disparity.convertTo(disparity, CV_32F, 1. / cv::StereoSGBM::DISP_SCALE);

		// calculate stixels
		std::vector<Stixel> stixels;

		const auto t1 = std::chrono::system_clock::now();

		stixelWorld.compute(disparity, stixels);

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "stixel computation time: " << 1e-3 * duration << "[msec]" << std::endl;

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(I1, draw, cv::COLOR_GRAY2BGRA);

		cv::Mat stixelImg = cv::Mat::zeros(I1.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp, (float)numDisparities));

		draw = draw + 0.5 * stixelImg;

		cv::imshow("disparity", disparity / numDisparities);
		cv::imshow("stixels", draw);

		const char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
	}
}