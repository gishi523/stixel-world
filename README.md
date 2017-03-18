# stixel-world
An implementation of stixel computation

====

![stixel-world](wiki/images/stixels.png)

## Description
- An implementation of the static Stixel computation based on [1,2].
- Extracts the static Stixels from the input disparity map.
- Not a dynamic Stixel. It means that tracking and estimating motion of each Stixel is not supported.

## References
- [1] D. Pfeiffer, U. Franke: “Efficient Representation of Traffic Scenes by means of Dynamic Stixels”, IEEE Intelligent Vehicles Symposium IV 2010, San Diego, CA, Juni 2010
- [2] H. Badino, U. Franke, and D. Pfeiffer, “The stixel world - a compact medium level representation of the 3d-world,” in DAGM Symposium, (Jena, Germany), September 2009.

## Demo
- <a href="https://www.youtube.com/watch?v=i8dcQYPC2kg" target="_blank">Demo1</a>
- <a href="https://www.youtube.com/watch?v=mQTMts0-njQ" target="_blank">Demo2</a>

## Requirement
- OpenCV

## How to build
```
$ git clone https://github.com/gishi523/stixel-world.git
$ cd stixel-world
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to use
```
./stixelworld left-image-format right-image-format camera.xml
```
- left-image-format
    - the left image sequence
- right-image-format
    - the right image sequence
- camera.xml
    - the camera intrinsic and extrinsic parameters

### Example
 ```
./stixelworld images/img_c0_%09d.pgm images/img_c1_%09d.pgm ../camera.xml
```

### Data
- I tested this work using the Daimler Ground Truth Stixel Dataset
- http://www.6d-vision.com/ground-truth-stixel-dataset

## Author
gishi523
