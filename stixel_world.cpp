#include "stixel_world.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using CameraParameters = StixelWorld::CameraParameters;

// Transformation between pixel coordinate and world coordinate
struct CoordinateTransform
{
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

// Implementation of road disparity estimation by RANSAC
class RoadEstimation
{
public:

	struct Parameters
	{
		int samplingStep;      //!< pixel step of disparity sampling
		int minDisparity;      //!< minimum disparity used for RANSAC

		int maxIterations;     //!< number of iterations of RANSAC
		float inlierRadius;    //!< inlier radius of RANSAC
		float maxCameraHeight; //!< maximum acceptable camera height

		// default settings
		Parameters()
		{
			samplingStep = 2;
			minDisparity = 10;

			maxIterations = 32;
			inlierRadius = 1;
			maxCameraHeight = 5;
		}
	};

	RoadEstimation(const Parameters& param = Parameters()) : param_(param)
	{
	}

	cv::Vec2f compute(const cv::Mat1f& disparity, const CameraParameters& camera)
	{
		CV_Assert(disparity.type() == CV_32F);

		const int umax = disparity.rows;
		const int vmax = disparity.cols;
		const int dmin = param_.minDisparity;

		// sample v-disparity points
		points_.reserve(vmax * umax);
		points_.clear();
		for (int u = 0; u < umax; u += param_.samplingStep)
		{
			for (int v = 0; v < vmax; v += param_.samplingStep)
			{
				const float d = disparity.ptr<float>(u)[v];
				if (d >= dmin)
					points_.push_back(cv::Point2f(static_cast<float>(v), d));
			}
		}
		if (points_.empty())
			return cv::Vec2f(0, 0);

		// estimate line by RANSAC
		const int npoints = static_cast<int>(points_.size());
		cv::RNG random;
		cv::Vec2f bestLine(0, 0);
		int maxInliers = 0;
		for (int iter = 0; iter < param_.maxIterations; iter++)
		{
			const cv::Point2f& pt1 = points_[random.next() % npoints];
			const cv::Point2f& pt2 = points_[random.next() % npoints];
			const float a = (pt2.y - pt1.y) / (pt2.x - pt1.x);
			const float b = -a * pt1.x + pt1.y;

			// estimate camera tilt and height
			const float tilt = atanf((a * camera.v0 + b) / (camera.fu * a));
			const float height = camera.baseline * cosf(tilt) / a;

			// skip if not within valid range
			if (height <= 0.f || height > param_.maxCameraHeight)
				continue;

			// count inliers within a radius and update the best line
			int inliers = 0;
			for (int i = 0; i < npoints; i++)
			{
				const float y = points_[i].x;
				const float x = points_[i].y;
				const float yhat = a * y + b;
				if (fabs(yhat - x) <= param_.inlierRadius)
					inliers++;
			}

			if (inliers > maxInliers)
			{
				maxInliers = inliers;
				bestLine = cv::Vec2f(a, b);
			}
		}

		// apply least squares fitting using inliers around the best line
		double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
		int n = 0;
		for (int i = 0; i < npoints; i++)
		{
			const float x = points_[i].x;
			const float y = points_[i].y;
			const float yhat = bestLine[0] * x + bestLine[1];
			if (fabs(yhat - y) <= param_.inlierRadius)
			{
				sx += x;
				sy += y;
				sxx += x * x;
				syy += y * y;
				sxy += x * y;
				n++;
			}
		}

		const float a = static_cast<float>((n * sxy - sx * sy) / (n * sxx - sx * sx));
		const float b = static_cast<float>((sxx * sy - sxy * sx) / (n * sxx - sx * sx));
		return cv::Vec2f(a, b);
	}

private:
	Parameters param_;
	std::vector<cv::Point2f> points_;
};

// Implementation of free space computation
class FreeSpace
{
public:

	struct Parameters
	{
		float alpha1;       //!< weight for object evidence
		float alpha2;       //!< weight for road evidence
		float objectHeight; //!< assumed object height
		float Cs;           //!< cost parameter penalizing jumps in depth
		float Ts;           //!< threshold saturating the cost function
		int maxPixelJump;   //!< maximum allowed jumps in pixel (higher value increases computation time)
		int mode;

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

	void compute(const cv::Mat1f& disparity, std::vector<float>& roadDisp, int vhor, std::vector<int>& path, const CameraParameters& camera)
	{
		const int umax = disparity.rows;
		const int vmax = disparity.cols;

		cv::Mat1f score(umax, vmax, std::numeric_limits<float>::max());
		cv::Mat1i table(umax, vmax, 0);

		CoordinateTransform tf(camera);

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

static cv::Vec2f calcRoadDisparityParams(const cv::Mat1f& columns, const CameraParameters& camera, int mode)
{
	if (mode == StixelWorld::ROAD_ESTIMATION_AUTO)
	{
		// estimate from v-disparity
		RoadEstimation roadEstimation;
		return roadEstimation.compute(columns, camera);
	}
	else if (mode == StixelWorld::ROAD_ESTIMATION_CAMERA)
	{
		// estimate from camera tilt and height
		const float sinTilt = sinf(camera.tilt);
		const float cosTilt = cosf(camera.tilt);
		const float a = (camera.baseline / camera.height) * cosTilt;
		const float b = (camera.baseline / camera.height) * (camera.fu * sinTilt - camera.v0 * cosTilt);
		return cv::Vec2f(a, b);
	}

	CV_Error(cv::Error::StsInternal, "No such mode");
	return cv::Vec2f(0, 0);
}

static float calcAverageDisparity(const cv::Mat& disparity, const cv::Rect& rect, int minDisp, int maxDisp)
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
	const int umax = disparity.cols / stixelWidth;
	const int vmax = disparity.rows;
	CameraParameters camera = param_.camera;

	// compute horizontal median of each column
	cv::Mat1f columns(umax, vmax);
	std::vector<float> buf(stixelWidth);
	for (int v = 0; v < vmax; v++)
	{
		for (int u = 0; u < umax; u++)
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

	// compute road model (assumes planar surface)
	const cv::Vec2f line = calcRoadDisparityParams(columns, camera, param_.roadEstimation);
	const float a = line[0];
	const float b = line[1];

	// compute expected road disparity
	std::vector<float> roadDisp(vmax);
	for (int v = 0; v < vmax; v++)
		roadDisp[v] = a * v + b;

	// horizontal row from which road dispaliry becomes negative
	const int vhor = cvRound(-b / a);

	// when AUTO mode, update camera tilt and height
	if (param_.roadEstimation == ROAD_ESTIMATION_AUTO)
	{
		camera.tilt = atanf((a * camera.v0 + b) / (camera.fu * a));
		camera.height = camera.baseline * cosf(camera.tilt) / a;
	}

	// free space computation
	FreeSpace freeSpace;
	freeSpace.compute(columns, roadDisp, vhor, lowerPath_, camera);

	// height segmentation
	HeightSegmentation heightSegmentation;
	heightSegmentation.compute(columns, lowerPath_, upperPath_, camera);

	// extract disparity
	for (int u = 0; u < umax; u++)
	{
		const int vT = upperPath_[u];
		const int vB = lowerPath_[u];
		const int stixelHeight = vB - vT;
		const cv::Rect stixelRegion(stixelWidth * u, vT, stixelWidth, stixelHeight);

		Stixel stixel;
		stixel.u = stixelWidth * u + stixelWidth / 2;
		stixel.vT = vT;
		stixel.vB = vB;
		stixel.width = stixelWidth;
		stixel.disp = calcAverageDisparity(disparity, stixelRegion, param_.minDisparity, param_.maxDisparity);
		stixels.push_back(stixel);
	}
}