# stixel-world
An implementation of stixel computation

====

![stixel-world](https://dl.dropboxusercontent.com/u/36775496/stixels.png)

## 概要
- Daimlerの論文[1,2]を参考にStixel Computationを実装したものです
- ステレオ視差画像からStixel(物体を覆う棒状の領域)を計算します

## 参考
- [1] D. Pfeiffer, U. Franke: “Efficient Representation of Traffic Scenes by means of Dynamic Stixels”, IEEE Intelligent Vehicles Symposium IV 2010, San Diego, CA, Juni 2010
- [2] H. Badino, U. Franke, and D. Pfeiffer, “The stixel world - a compact medium level representation of the 3d-world,” in DAGM Symposium, (Jena, Germany), September 2009.

## Requirement
- OpenCVが必要です

## ビルド
```
$ git clone https://github.com/gishi523/stixel-world.git
$ cd stixel-world
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## 使い方
```
./stixelworld left-image-format right-image-format camera.xml
```
- left-image-format
    - 左側の連番画像
- left-image-format
    - 右側の連番画像
- camera.xml
    - 計算に必要なカメラパラメータ
    - シーンに応じて設定して下さい

### 実行例
```
./stixelworld images/img_c0_%09d.pgm images/img_c1_%09d.pgm ../camera.xml
```

### データ
- DaimlerのGround Truth Stixel Datasetに含まれる画像およびカメラパラメータで動作を確認しています
- http://www.6d-vision.com/ground-truth-stixel-dataset

## Author
gishi523
