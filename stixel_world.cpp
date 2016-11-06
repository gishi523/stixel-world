#include "stixel_world.h"

static void computeFreeSpace(const cv::Mat& disp, std::vector<int>& path, float paramO, float paramR,
	float focalLengthX, float focalLengthY, float principalPointX, float principalPointY,
	float baseline, float cameraHeight, float cameraTilt)
{
	CV_Assert(disp.type() == CV_32F);

	// transpose for efficient memory access
	cv::Mat1f dispt = disp.t();

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

	for (int u = 0; u < umax; u++)
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
			float roadscore = integralRoadDiff[vmax - 1] - integralRoadDiff[vb - 1];
			//CV_Assert(roadscore >= 0.f);

			score(u, vb) = paramO * objectscore + paramR * roadscore;
		}
	}

	// extract the optimal free space path
	cv::Mat1s table = cv::Mat1s::zeros(score.size());
	const int maxpixjumb = 50;
	const float P1 = 50.f;
	const float P2 = 32.f;

	// forward path
	for (int u = 1; u < umax; u++)
	{
		for (int v = vhori; v < vmax; v++)
		{
			float minscore = FLT_MAX;
			int minpath = 0;

			int vvt = std::max(v - maxpixjumb, vhori);
			int vvb = std::min(v + maxpixjumb + 1, vmax);

			float d = dispt(u, v);
			for (int vv = vvt; vv < vvb; vv++)
			{
				float dd = dispt(u - 1, vv);
				float dispjump = (d >= 0.f && dd >= 0.f) ? fabsf(dd - d) : SCORE_DEFAULT;
				float penalty = std::min(P1 * dispjump, P1 * P2);
				float s = score(u - 1, vv) + penalty;
				if (s < minscore)
				{
					minscore = s;
					minpath = vv;
				}
			}

			score(u, v) += minscore;
			table(u, v) = minpath;
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

StixelWrold::StixelWrold(
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

void StixelWrold::compute(const cv::Mat& disp, std::vector<Stixel>& stixels, int stixelWidth)
{
	CV_Assert(disp.type() == CV_32F);

	// free space computation
	computeFreeSpace(disp, lowerPath, 2.f, 1.f, focalLengthX_, focalLengthY_, principalPointX_, principalPointY_,
		baseline_, cameraHeight_, cameraTilt_);

	// height segmentation

	// extract stixels
}