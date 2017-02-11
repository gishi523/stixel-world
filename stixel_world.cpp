#include "stixel_world.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static void reduceDisparity(const cv::Mat& src, cv::Mat& dst, int stixelWidth)
{
	dst.create(cv::Size(src.cols / stixelWidth, src.rows), CV_32F);
	std::vector<float> buf(stixelWidth);
	for (int v = 0; v < src.rows; v++)
	{
		for (int us = 0, ud = 0; ud < dst.cols; us += stixelWidth, ud++)
		{
			for (int i = 0; i < stixelWidth; i++)
				buf[i] = src.at<float>(v, us + i);

			std::sort(std::begin(buf), std::end(buf));

			dst.at<float>(v, ud) = buf[stixelWidth / 2];
		}
	}
}

static void computeFreeSpace(const cv::Mat& disp, std::vector<int>& path, float paramO, float paramR,
	float focalLengthX, float focalLengthY, float principalPointX, float principalPointY,
	float baseline, float cameraHeight, float cameraTilt)
{
	CV_Assert(disp.type() == CV_32F);

	// transpose for efficient memory access
	const cv::Mat1f dispt = disp.t();

	const int umax = dispt.rows;
	const int vmax = dispt.cols;

	const float sinTilt = sinf(cameraTilt);
	const float cosTilt = cosf(cameraTilt);

	// calculate road disparity
	std::vector<float> roadDisp(vmax);
	for (int v = 0; v < vmax; v++)
		roadDisp[v] = (baseline / cameraHeight) * (focalLengthX * sinTilt + (v - principalPointY) * cosTilt);

	// search horizontal row (row from which road dispaliry becomes negative)
	int vhori = 0;
	for (int v = vmax - 1; v >= 0; v--)
	{
		if (roadDisp[v] < 0.f)
		{
			vhori = v + 1;
			break;
		}
	}

	// calculate score image
	const float SCORE_INV = -1.f;
	const float SCORE_DEFAULT = 1.f;
	cv::Mat1f score(dispt.size());

	for (int vb = 0; vb < vhori; vb++)
		score.col(vb) = SCORE_INV;

	int u;
#pragma omp parallel for
	for (u = 0; u < umax; u++)
	{
		// compute and accumlate differences between measured disparity and road disparity
		std::vector<float> integralRoadDiff(vmax);
		float integralRoadDiffOld = 0.f;
		for (int v = vhori; v < vmax; v++)
		{
			float roadDiff = dispt(u, v) > 0.f ? fabsf(dispt(u, v) - roadDisp[v]) : SCORE_DEFAULT;
			integralRoadDiff[v] = integralRoadDiffOld + roadDiff;
			integralRoadDiffOld = integralRoadDiff[v];
		}

		// compute search range
		std::vector<int> vtop(vmax, 0);
		const float objectHeight = 0.5f;
		for (int vb = vhori; vb < vmax; vb++)
		{
			const float coef = (baseline / roadDisp[vb]) * (focalLengthX / focalLengthY);
			const float Yb = coef * ((vb - principalPointY) * cosTilt + focalLengthY * sinTilt);
			const float Zb = coef * (focalLengthY * cosTilt - (vb - principalPointY) * sinTilt);
			const float Yt = Yb - objectHeight;
			const float vt = focalLengthY * (Yt * cosTilt - Zb * sinTilt) / (Yt * sinTilt + Zb * cosTilt) + principalPointY;
			vtop[vb] = std::max(cvRound(vt), 0);
		}

		for (int vb = vhori; vb < vmax; vb++)
		{
			// calculate the object score
			float objectscore = 0.f;
			for (int v = vtop[vb]; v < vb; ++v)
				objectscore += dispt(u, v) > 0.f ? fabsf(dispt(u, v) - roadDisp[vb]) : SCORE_DEFAULT;

			// calculate the road score
			const float roadscore = integralRoadDiff[vmax - 1] - integralRoadDiff[vb - 1];
			//CV_Assert(roadscore >= 0.f);

			score(u, vb) = paramO * objectscore + paramR * roadscore;
		}
	}

	// extract the optimal free space path
	cv::Mat1s table = cv::Mat1s::zeros(score.size());
	const float P1 = 50.f;
	const float P2 = 32.f;
	const int maxpixjumb = 100;

	// forward path
	for (int u = 1; u < umax; u++)
	{
		int v;
#pragma omp parallel for
		for (v = vhori; v < vmax; v++)
		{
			float minscore = FLT_MAX;
			int minv = 0;

			const int vvt = std::max(v - maxpixjumb, vhori);
			const int vvb = std::min(v + maxpixjumb + 1, vmax);

			const float d = dispt(u, v);
			for (int vv = vvt; vv < vvb; vv++)
			{
				const float dd = dispt(u - 1, vv);
				const float dispjump = (d >= 0.f && dd >= 0.f) ? fabsf(dd - d) : SCORE_DEFAULT;
				const float penalty = std::min(P1 * dispjump, P1 * P2);
				const float s = score(u - 1, vv) + penalty;
				if (s < minscore)
				{
					minscore = s;
					minv = vv;
				}
			}

			score(u, v) += minscore;
			table(u, v) = minv;
		}
	}

	// backward path
	path.resize(umax);
	float minscore = FLT_MAX;
	int minv = 0;
	for (int v = vhori; v < vmax; v++)
	{
		if (score(umax - 1, v) < minscore)
		{
			minscore = score(umax - 1, v);
			minv = v;
		}
	}
	for (int u = umax - 1; u >= 0; u--)
	{
		path[u] = minv;
		minv = table(u, minv);
	}
}

static void heightSegmentation(const cv::Mat& disp, const std::vector<int>& lowerPath, std::vector<int>& upperPath,
	float focalLengthX, float focalLengthY, float principalPointX, float principalPointY,
	float baseline, float cameraHeight, float cameraTilt)
{
	CV_Assert(disp.type() == CV_32F);

	// transpose for efficient memory access
	const cv::Mat1f dispt = disp.t();

	const int umax = dispt.rows;
	const int vmax = dispt.cols;

	const float sinTilt = sinf(cameraTilt);
	const float cosTilt = cosf(cameraTilt);

	cv::Mat1f score(dispt.size());

	int u;
#pragma omp parallel for
	for (u = 0; u < umax; u++)
	{
		// compute and accumlate membership value
		std::vector<float> integralMembership(vmax);
		float integralMembershipOld = 0.f;

		const int vb = lowerPath[u];
		const float deltaZ = 5.f;
		const float db = dispt(u, vb);

		float deltaD = 0.f;
		if (db > 0.f)
		{
			const float coef = (baseline / db) * (focalLengthX / focalLengthY);
			const float Yb = coef * ((vb - principalPointY) * cosTilt + focalLengthY * sinTilt);
			const float Zb = coef * (focalLengthY * cosTilt - (vb - principalPointY) * sinTilt);
			const float Zb_deltaZ = Zb + deltaZ;
			const float db_deltaD = baseline * focalLengthX / (Yb * sinTilt + Zb_deltaZ * cosTilt);
			deltaD = db_deltaD - db;
		}

		for (int v = 0; v < vmax; v++)
		{
			const float d = dispt(u, v);

			float membership = 0.f;
			if (db > 0.f && d > 0.f)
			{
				const float deltad = (d - db) / deltaD;
				const float exponent = 1.f - deltad * deltad;
				membership = powf(2.f, exponent) - 1.f;
			}

			integralMembership[v] = integralMembershipOld + membership;
			integralMembershipOld = integralMembership[v];
		}

		score(u, 0) = integralMembership[vb - 1];
		for (int vh = 1; vh < vb; vh++)
		{
			const float score1 = integralMembership[vh - 1];
			const float score2 = integralMembership[vb - 1] - integralMembership[vh - 1];
			score(u, vh) = score1 - score2;
		}
	}

	// extract the optimal free space path
	cv::Mat1s table = cv::Mat1s::zeros(score.size());
	const float Cs = 8;
	const float Nz = 5;
	const int maxpixjumb = 50;

	// forward upperpath
	for (int u = 1; u < umax; u++)
	{
		const int vb = lowerPath[u];
		int vc;
#pragma omp parallel for
		for (vc = 0; vc < vb; vc++)
		{
			const float dc = dispt(u, vc);
			const int vpt = std::max(vc - maxpixjumb, 0);
			const int vpb = std::min(vc + maxpixjumb + 1, vb);

			float minscore = FLT_MAX;
			int minv = 0;

			for (int vp = vpt; vp < vpb; vp++)
			{
				const float dp = dispt(u - 1, vp);

				float Cz = 1.f;
				if (dc > 0.f && dp > 0.f)
				{
					const float coefc = (baseline / dc) * (focalLengthX / focalLengthY);
					const float Zc = coefc * (focalLengthY * cosTilt - (vc - principalPointY) * sinTilt);

					const float coefp = (baseline / dp) * (focalLengthX / focalLengthY);
					const float Zp = coefp * (focalLengthY * cosTilt - (vp - principalPointY) * sinTilt);

					Cz = std::max(0.f, 1 - fabsf(Zc - Zp) / Nz);
				}

				const float penalty = Cs * abs(vc - vp) * Cz;

				const float s = score(u - 1, vp) + penalty;
				if (s < minscore)
				{
					minscore = s;
					minv = vp;
				}
			}

			score(u, vc) += minscore;
			table(u, vc) = minv;
		}
	}

	// backward
	upperPath.resize(umax);
	float minscore = FLT_MAX;
	int minv = 0;
	for (int v = 0; v < vmax; v++)
	{
		if (score(umax - 1, v) < minscore)
		{
			minscore = score(umax - 1, v);
			minv = v;
		}
	}
	for (int u = umax - 1; u >= 0; u--)
	{
		upperPath[u] = minv;
		minv = table(u, minv);
	}
}

static float extractDisparity(const cv::Mat& disp, const cv::Rect& area, int maxDisp)
{
	const cv::Rect imageArea(0, 0, disp.cols, disp.rows);
	const cv::Mat roi(disp, area & imageArea);

	const int histSize[] = { maxDisp + 1 };
	const float range[] = { -1, static_cast<float>(maxDisp) };
	const float* ranges[] = { range };

	cv::Mat hist;
	cv::calcHist(&roi, 1, 0, cv::Mat(), hist, 1, histSize, ranges);

	double maxVal;
	int maxIdx;
	cv::minMaxIdx(hist, NULL, &maxVal, NULL, &maxIdx);

	const double ret = (range[1] - range[0]) * maxIdx / histSize[0] + range[0];
	return static_cast<float>(ret);
}

static void extractStixel(const cv::Mat& disp, const std::vector<int>& lowerPath, const std::vector<int>& upperPath,
	std::vector<Stixel>& stixels, int stixelWidth)
{
	CV_Assert(stixelWidth % 2 == 1);

	double maxDisp;
	cv::minMaxLoc(disp, NULL, &maxDisp);

	for (int u = 0; u < (int)upperPath.size(); u++)
	{
		const int vT = upperPath[u];
		const int vB = lowerPath[u];
		const int stixelHeight = vB - vT;
		const cv::Rect stixelArea(stixelWidth * u, vT, stixelWidth, stixelHeight);

		Stixel stixel;
		stixel.u = stixelWidth * u + stixelWidth / 2;
		stixel.vT = vT;
		stixel.vB = vB;
		stixel.width = stixelWidth;
		stixel.disp = extractDisparity(disp, stixelArea, static_cast<int>(maxDisp));
		stixels.push_back(stixel);
	}
}

StixelWorld::StixelWorld(
	float focalLengthX,
	float focalLengthY,
	float principalPointX,
	float principalPointY,
	float baseline,
	float cameraHeight,
	float cameraTilt)
	: focalLengthX_(focalLengthX), focalLengthY_(focalLengthY),
	principalPointX_(principalPointX), principalPointY_(principalPointY),
	baseline_(baseline), cameraHeight_(cameraHeight), cameraTilt_(cameraTilt)
{

}

void StixelWorld::compute(const cv::Mat& disp, std::vector<Stixel>& stixels, int stixelWidth)
{
	CV_Assert(disp.type() == CV_32F);

	// reduce disparity
	cv::Mat dispr(cv::Size(disp.cols / stixelWidth, disp.rows), CV_32F);
	reduceDisparity(disp, dispr, stixelWidth);

	// free space computation
	computeFreeSpace(dispr, lowerPath, 2.f, 1.f, focalLengthX_, focalLengthY_, principalPointX_, principalPointY_,
		baseline_, cameraHeight_, cameraTilt_);

	// height segmentation
	heightSegmentation(dispr, lowerPath, upperPath, focalLengthX_, focalLengthY_, principalPointX_, principalPointY_,
		baseline_, cameraHeight_, cameraTilt_);

	// extract stixels
	extractStixel(disp, lowerPath, upperPath, stixels, stixelWidth);
}
