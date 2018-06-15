#ifndef __STIXEL_WORLD_H__
#define __STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>

/** @brief Stixel struct
*/
struct Stixel
{
	int u;                        //!< stixel center x position
	int vT;                       //!< stixel top y position
	int vB;                       //!< stixel bottom y position
	int width;                    //!< stixel width
	float disp;                   //!< stixel average disparity
};

/** @brief StixelWorld class.

The class implements the static Stixel computation based on [1,2].
[1] D. Pfeiffer, U. Franke: "Efficient Representation of Traffic Scenes by means of Dynamic Stixels"
[2] H. Badino, U. Franke, and D. Pfeiffer, "The stixel world - a compact medium level representation of the 3d-world,"
*/
class StixelWorld
{
public:

	enum
	{
		ROAD_ESTIMATION_AUTO = 0, //!< road disparity are estimated by input disparity
		ROAD_ESTIMATION_CAMERA    //!< road disparity are estimated by camera tilt and height
	};

	/** @brief CameraParameters struct
	*/
	struct CameraParameters
	{
		float fu;                 //!< focal length x (pixel)
		float fv;                 //!< focal length y (pixel)
		float u0;                 //!< principal point x (pixel)
		float v0;                 //!< principal point y (pixel)
		float baseline;           //!< baseline (meter)
		float height;             //!< height position (meter), ignored when ROAD_ESTIMATION_AUTO
		float tilt;               //!< tilt angle (radian), ignored when ROAD_ESTIMATION_AUTO

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

	/** @brief Parameters struct
	*/
	struct Parameters
	{
		int stixelWidth;          //!< stixel width, must be odd number
		int minDisparity;         //!< minimum value of input disparity (half-open interval [min, max))
		int maxDisparity;         //!< maximum value of input disparity (half-open interval [min, max))
		int roadEstimation;       //!< road disparity estimation mode
		CameraParameters camera;  //!< camera parameters

		// default settings
		Parameters()
		{
			stixelWidth = 7;
			minDisparity = -1;
			maxDisparity = 64;
			roadEstimation = ROAD_ESTIMATION_AUTO;
			camera = CameraParameters();
		}
	};

	/** @brief The constructor
	@param param input parameters
	*/
	StixelWorld(const Parameters& param = Parameters());

	/** @brief Computes stixels in a disparity map
	@param disparity 32-bit single-channel disparity map
	@param output array of stixels
	*/
	void compute(const cv::Mat& disparity, std::vector<Stixel>& stixels);

private:
	std::vector<int> lowerPath_, upperPath_;
	Parameters param_;
};

#endif // !__STIXEL_WORLD_H__