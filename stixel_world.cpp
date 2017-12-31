#include "stixel_world.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Transformation between pixel coordinate and world coordinate
struct CoordinateTransform
{
	using CameraParameters = StixelWorld::CameraParameters;

	CoordinateTransform(const CameraParameters& camera) : camera(camera)
	{
		sinTilt = (sinf(camera.tilt));
		cosTilt = (cosf(camera.tilt));
		B = camera.baseline * camera.fu / camera.fv;
	}

	inline float toY(float d, int v) const
	{
		return (B / d) * ((v - camera.v0) * cosTilt + camera.fv * sinTilt);
	}

	inline float toZ(float d, int v) const
	{
		return (B / d) * (camera.fv * cosTilt - (v - camera.v0) * sinTilt);
	}

	inline float toV(float Y, float Z) const
	{
		return camera.fv * (Y * cosTilt - Z * sinTilt) / (Y * sinTilt + Z * cosTilt) + camera.v0;
	}

	inline float toD(float Y, float Z) const
	{
		return camera.baseline * camera.fu / (Y * sinTilt + Z * cosTilt);
	}

	CameraParameters camera;
	float sinTilt, cosTilt, B;
};

// Implementation of free space computation
class FreeSpace
{
public:

	using CameraParameters = StixelWorld::CameraParameters;

	struct Parameters
	{
		float alpha1;       //!< weight for object evidence
		float alpha2;       //!< weight for road evidence
		float objectHeight; //!< assumed object height
		float Cs;           //!< cost parameter penalizing jumps in depth
		float Ts;           //!< threshold saturating the cost function
		int maxPixelJump;   //!< maximum allowed jumps in pixel (higher value increases computation time)

		// default settings
		Parameters()
		{
			alpha1 = 2;
			alpha2 = 1;
			objectHeight = 0.5f;
			Cs = 50;
			Ts = 32;
			maxPixelJump = 100;
		}
	};

	FreeSpace(const Parameters& param = Parameters()) : param_(param)
	{
	}

	void compute(const cv::Mat1f& disparity, std::vector<int>& path, const CameraParameters& camera)
	{
		const int umax = disparity.rows;
		const int vmax = disparity.cols;

		cv::Mat1f score(umax, vmax, std::numeric_limits<float>::max());
		cv::Mat1i table(umax, vmax, 0);
		
		CoordinateTransform tf(camera);

		/////////////////////////////////////////////////////////////////////////////
		// compute expected road disparity
		//////////////////////////////////////////////////////////////////////////////
		const float sinTilt = sinf(camera.tilt);
		const float cosTilt = cosf(camera.tilt);

		// assumes planar surface
		std::vector<float> roadDisp(vmax);
		for (int v = 0; v < vmax; v++)
			roadDisp[v] = (camera.baseline / camera.height) * (camera.fu * sinTilt + (v - camera.v0) * cosTilt);

		// get horizontal row (row from which road dispaliry becomes negative)
		const int vhor = cvRound((camera.v0 * cosTilt - camera.fu * sinTilt) / cosTilt);

		/////////////////////////////////////////////////////////////////////////////
		// compute score image for the free space
		//////////////////////////////////////////////////////////////////////////////
		const float SCORE_DEFAULT = 1.f;

		int u;
#pragma omp parallel for
		for (u = 0; u < umax; u++)
		{
			// compute and accumlate differences between measured disparity and expected road disparity
			std::vector<float> integralRoadDiff(vmax);
			float tmpSum = 0.f;
			for (int v = vhor; v < vmax; v++)
			{
				const float roadDiff = disparity(u, v) > 0.f ? fabsf(disparity(u, v) - roadDisp[v]) : SCORE_DEFAULT;
				tmpSum += roadDiff;
				integralRoadDiff[v] = tmpSum;
			}

			// compute search range
			std::vector<int> vT(vmax, 0);
			for (int vB = vhor; vB < vmax; vB++)
			{
				const float YB = tf.toY(roadDisp[vB], vB);
				const float ZB = tf.toZ(roadDisp[vB], vB);
				const float YT = YB - param_.objectHeight;
				vT[vB] = std::max(cvRound(tf.toV(YT, ZB)), 0);
			}

			for (int vB = vhor; vB < vmax; vB++)
			{
				// compute the object score
				float objectScore = 0.f;
				for (int v = vT[vB]; v < vB; ++v)
					objectScore += disparity(u, v) > 0.f ? fabsf(disparity(u, v) - roadDisp[vB]) : SCORE_DEFAULT;

				// compute the road score
				const float roadScore = integralRoadDiff[vmax - 1] - integralRoadDiff[vB - 1];

				score(u, vB) = param_.alpha1 * objectScore + param_.alpha2 * roadScore;
			}
		}

		/////////////////////////////////////////////////////////////////////////////
		// extract the optimal free space path by dynamic programming
		//////////////////////////////////////////////////////////////////////////////
		// forward step
		for (int uc = 1; uc < umax; uc++)
		{
			const int up = uc - 1;

			int vc;
#pragma omp parallel for
			for (vc = vhor; vc < vmax; vc++)
			{
				const int vp1 = std::max(vc - param_.maxPixelJump, vhor);
				const int vp2 = std::min(vc + param_.maxPixelJump + 1, vmax);

				float minScore = std::numeric_limits<float>::max();
				int minv = 0;
				for (int vp = vp1; vp < vp2; vp++)
				{
					const float dc = disparity(uc, vc);
					const float dp = disparity(up, vp);
					const float dispJump = (dc >= 0.f && dp >= 0.f) ? fabsf(dp - dc) : SCORE_DEFAULT;
					const float penalty = std::min(param_.Cs * dispJump, param_.Cs * param_.Ts);
					const float s = score(up, vp) + penalty;
					if (s < minScore)
					{
						minScore = s;
						minv = vp;
					}
				}

				score(uc, vc) += minScore;
				table(uc, vc) = minv;
			}
		}

		// backward step
		path.resize(umax);
		float minScore = std::numeric_limits<float>::max();
		int minv = 0;
		for (int v = vhor; v < vmax; v++)
		{
			if (score(umax - 1, v) < minScore)
			{
				minScore = score(umax - 1, v);
				minv = v;
			}
		}
		for (int u = umax - 1; u >= 0; u--)
		{
			path[u] = minv;
			minv = table(u, minv);
		}
	}

private:
	Parameters param_;
};

// Implementation of height segmentation
class HeightSegmentation
{
public:

	using CameraParameters = StixelWorld::CameraParameters;

	struct Parameters
	{
		float deltaZ;     //!< allowed deviation in [m] to the base point
		float Cs;         //!< cost parameter penalizing jumps in depth and pixel
		float Nz;         //!< if the difference in depth between the columns is equal or larger than this value, cost of a jump becomes zero
		int maxPixelJump; //!< maximum allowed jumps in pixel (higher value increases computation time)

		// default settings
		Parameters()
		{
			deltaZ = 5;
			Cs = 8;
			Nz = 5;
			maxPixelJump = 50;
		}
	};

	HeightSegmentation(const Parameters& param = Parameters()) : param_(param)
	{
	}

	void compute(const cv::Mat1f& disparity, const std::vector<int>& lowerPath, std::vector<int>& upperPath, const CameraParameters& camera)
	{
		const int umax = disparity.rows;
		const int vmax = disparity.cols;

		cv::Mat1f score(umax, vmax, std::numeric_limits<float>::max());
		cv::Mat1i table(umax, vmax, 0);
		
		CoordinateTransform tf(camera);

		/////////////////////////////////////////////////////////////////////////////
		// compute score image for the height segmentation
		//////////////////////////////////////////////////////////////////////////////
		int u;
#pragma omp parallel for
		for (u = 0; u < umax; u++)
		{
			// get the base point
			const int vB = lowerPath[u];
			const float dB = disparity(u, vB);

			// deltaD represents the allowed deviation in disparity
			float deltaD = 0.f;
			if (dB > 0.f)
			{
				const float YB = tf.toY(dB, vB);
				const float ZB = tf.toZ(dB, vB);
				deltaD = dB - tf.toD(YB, ZB + param_.deltaZ);
			}

			// compute and accumlate membership value
			std::vector<float> integralMembership(vmax);
			float tmpSum = 0.f;
			for (int v = 0; v < vmax; v++)
			{
				const float d = disparity(u, v);

				float membership = 0.f;
				if (dB > 0.f && d > 0.f)
				{
					const float deltad = (d - dB) / deltaD;
					const float exponent = 1.f - deltad * deltad;
					membership = powf(2.f, exponent) - 1.f;
				}

				tmpSum += membership;
				integralMembership[v] = tmpSum;
			}

			score(u, 0) = integralMembership[vB - 1];
			for (int vT = 1; vT < vB; vT++)
			{
				const float score1 = integralMembership[vT - 1];
				const float score2 = integralMembership[vB - 1] - integralMembership[vT - 1];
				score(u, vT) = score1 - score2;
			}
		}

		/////////////////////////////////////////////////////////////////////////////
		// extract the optimal height path by dynamic programming
		//////////////////////////////////////////////////////////////////////////////
		// forward step
		for (int uc = 1; uc < umax; uc++)
		{
			const int up = uc - 1;
			const int vB = lowerPath[uc];

			int vc;
#pragma omp parallel for
			for (vc = 0; vc < vB; vc++)
			{
				const int vp1 = std::max(vc - param_.maxPixelJump, 0);
				const int vp2 = std::min(vc + param_.maxPixelJump + 1, vB);

				float minScore = std::numeric_limits<float>::max();
				int minv = 0;
				for (int vp = vp1; vp < vp2; vp++)
				{
					const float dc = disparity(uc, vc);
					const float dp = disparity(up, vp);

					float Cz = 1.f;
					if (dc > 0.f && dp > 0.f)
					{
						const float Zc = tf.toZ(dc, vc);
						const float Zp = tf.toZ(dp, vp);
						Cz = std::max(0.f, 1 - fabsf(Zc - Zp) / param_.Nz);
					}

					const float penalty = param_.Cs * abs(vc - vp) * Cz;
					const float s = score(up, vp) + penalty;
					if (s < minScore)
					{
						minScore = s;
						minv = vp;
					}
				}

				score(uc, vc) += minScore;
				table(uc, vc) = minv;
			}
		}

		// backward step
		upperPath.resize(umax);
		float minScore = std::numeric_limits<float>::max();
		int minv = 0;
		for (int v = 0; v < vmax; v++)
		{
			if (score(umax - 1, v) < minScore)
			{
				minScore = score(umax - 1, v);
				minv = v;
			}
		}
		for (int u = umax - 1; u >= 0; u--)
		{
			upperPath[u] = minv;
			minv = table(u, minv);
		}
	}

private:
	Parameters param_;
};

static float averageDisparity(const cv::Mat& disparity, const cv::Rect& rect, int minDisp, int maxDisp)
{
	const cv::Mat dispROI = disparity(rect & cv::Rect(0, 0, disparity.cols, disparity.rows));
	const int histSize[] = { maxDisp - minDisp };
	const float range[] = { static_cast<float>(minDisp), static_cast<float>(maxDisp) };
	const float* ranges[] = { range };

	cv::Mat hist;
	cv::calcHist(&dispROI, 1, 0, cv::Mat(), hist, 1, histSize, ranges);

	int maxIdx[2];
	cv::minMaxIdx(hist, NULL, NULL, NULL, maxIdx);

	return (range[1] - range[0]) * maxIdx[0] / histSize[0] + range[0];
}

StixelWorld::StixelWorld(const Parameters & param) : param_(param)
{
}

void StixelWorld::compute(const cv::Mat& disparity, std::vector<Stixel>& stixels)
{
	CV_Assert(disparity.type() == CV_32F);
	CV_Assert(param_.stixelWidth % 2 == 1);

	const int stixelWidth = param_.stixelWidth;
	const int w = disparity.cols / stixelWidth;
	const int h = disparity.rows;

	// compute horizontal median of each column
	cv::Mat1f columns(w, h);
	std::vector<float> buf(stixelWidth);
	for (int v = 0; v < h; v++)
	{
		for (int u = 0; u < w; u++)
		{
			// compute horizontal median
			for (int du = 0; du < stixelWidth; du++)
				buf[du] = disparity.at<float>(v, u * stixelWidth + du);
			std::sort(std::begin(buf), std::end(buf));
			const float m = buf[stixelWidth / 2];

			// store with transposed
			columns.ptr<float>(u)[v] = m;
		}
	}

	// free space computation
	FreeSpace freeSpace;
	freeSpace.compute(columns, lowerPath, param_.camera);

	// height segmentation
	HeightSegmentation heightSegmentation;
	heightSegmentation.compute(columns, lowerPath, upperPath, param_.camera);

	// extract disparity
	for (int u = 0; u < w; u++)
	{
		const int vT = upperPath[u];
		const int vB = lowerPath[u];
		const int stixelHeight = vB - vT;
		const cv::Rect stixelRegion(stixelWidth * u, vT, stixelWidth, stixelHeight);

		Stixel stixel;
		stixel.u = stixelWidth * u + stixelWidth / 2;
		stixel.vT = vT;
		stixel.vB = vB;
		stixel.width = stixelWidth;
		stixel.disp = averageDisparity(disparity, stixelRegion, param_.minDisparity, param_.maxDisparity);
		stixels.push_back(stixel);
	}
}