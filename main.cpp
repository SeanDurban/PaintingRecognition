#include "Utilities.h"
#include <iostream>
#include <fstream>
#define NO_GALLERYS 4
#define NO_PAINTINGS 6

using namespace std;
vector<vector<vector<Point>>> groundTruths;
vector<vector<String>> groundTruthStrings;
vector<double> dice;
Mat gtImage;

Mat kmeans_clustering(Mat& image, int k, int iterations)
{
	CV_Assert(image.type() == CV_8UC3);
	// Populate an n*3 array of float for each of the n pixels in the image
	Mat samples(image.rows*image.cols, image.channels(), CV_32F);
	float* sample = samples.ptr<float>(0);
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				samples.at<float>(row*image.cols + col, channel) =
				(uchar)image.at<Vec3b>(row, col)[channel];
	// Apply k-means clustering to cluster all the samples so that each sample
	// is given a label and each label corresponds to a cluster with a particular
	// centre.
	Mat labels;
	Mat centres;
	kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1, 0.0001),
		iterations, KMEANS_PP_CENTERS, centres);
	// Put the relevant cluster centre values into a result image
	Mat& result_image = Mat(image.size(), image.type());
	for (int row = 0; row < image.rows; row++)
		for (int col = 0; col < image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				result_image.at<Vec3b>(row, col)[channel] = (uchar)centres.at<float>(*(labels.ptr<int>(row*image.cols + col)), channel);
	return result_image;
}

void hist(Mat& src)
{
	Mat hsv;
	cvtColor(src, hsv, CV_BGR2HSV);

	// Quantize the hue to 30 levels
	// and the saturation to 32 levels
	int hbins = 30, sbins = 32;
	int histSize[] = { hbins, sbins };
	// hue varies from 0 to 179, see cvtColor
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	MatND hist;
	// we compute the histogram from the 0-th and 1-st channels
	int channels[] = { 0, 1 };

	calcHist(&hsv, 1, channels, Mat(), // do not use mask
		hist, 2, histSize, ranges,
		true, // the histogram is uniform
		false);
	double maxVal = 0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);

	int scale = 10;
	Mat histImg = Mat::zeros(sbins*scale, hbins * 10, CV_8UC3);

	for (int h = 0; h < hbins; h++)
	{
		for (int s = 0; s < sbins; s++)
		{
			float binVal = hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(histImg, Point(h*scale, s*scale),
				Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				Scalar::all(intensity),
				CV_FILLED);
		}
}
	//int min, max, minP, maxP;
	//cvGetMinMaxHistValue(hist, min*, max*, minP*, maxP*);

	//namedWindow("Source", 1);
	//imshow("Source", src);

	namedWindow("H-S Histogram", 1);
	imshow("H-S Histogram", histImg);
	waitKey();
}



static MatND ComputeHistogram(Mat& image)
{

	Mat mImage = image.clone();
	int mNumberChannels;
	int* mChannelNumbers;
	int* mNumberBins;
	float mChannelRange[2];
	MatND mHistogram;
	mNumberChannels = mImage.channels();
	mChannelNumbers = new int[mNumberChannels];
	mNumberBins = new int[mNumberChannels];
	for (int count = 0; count<mNumberChannels; count++)
	{
		mChannelNumbers[count] = count;
		mNumberBins[count] = 20;
	}

	int mMinimumSaturation = 25;
	int mMinimumValue = 25;
	int mMaximumValue = 230;
	mChannelRange[0] = 0.0;
	mChannelRange[1] = 180.0;
	Mat hsv_image, hue_image, mask_image;
	cvtColor(mImage, hsv_image, CV_BGR2HSV);
	inRange(hsv_image, Scalar(0, mMinimumSaturation, mMinimumValue), Scalar(180, 256, mMaximumValue), mask_image);
	int channels[] = { 0,0 };
	hue_image.create(mImage.size(), mImage.depth());
	mixChannels(&hsv_image, 1, &hue_image, 1, channels, 1);
	const float* channel_ranges = mChannelRange;
	calcHist(&hue_image, 1, 0, mask_image, mHistogram, 1, mNumberBins, &channel_ranges);
	Mat display_image = Mat::zeros(mImage.size(), CV_8UC3);
	Draw1DHistogram(&mHistogram, 1, display_image);
	imshow("histogram", display_image);
	return mHistogram;
//	waitKey();
}

static void Draw1DHistogram(MatND histograms[], int number_of_histograms, Mat& display_image)
{
	int number_of_bins = histograms[0].size[0];
	double max_value = 0, min_value = 0;
	double channel_max_value = 0, channel_min_value = 0;
	for (int channel = 0; (channel < number_of_histograms); channel++)
	{
		minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
		max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
		min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
	}
	float scaling_factor = ((float)256.0) / ((float)number_of_bins);

	Mat histogram_image((int)(((float)number_of_bins)*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) + 1, CV_8UC3, Scalar(255, 255, 255));
	display_image = histogram_image;
	line(histogram_image, Point(0, 0), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
	line(histogram_image, Point(histogram_image.cols - 1, histogram_image.rows - 1), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
	int highest_point = static_cast<int>(0.9*((float)number_of_bins)*scaling_factor);
	for (int channel = 0; (channel < number_of_histograms); channel++)
	{
		int last_height;
		for (int h = 0; h < number_of_bins; h++)
		{
			float value = histograms[channel].at<float>(h);
			int height = static_cast<int>(value*highest_point / max_value);
			int where = (int)(((float)h)*scaling_factor);
			if (h > 0)
				line(histogram_image, Point((int)(((float)(h - 1))*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) - last_height),
					Point((int)(((float)h)*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) - height),
					Scalar(channel == 0 ? 255 : 0, channel == 1 ? 255 : 0, channel == 2 ? 255 : 0));
			last_height = height;
		}
	}
}
static void houghLinesApproach(Mat& image)
{
	Mat hsv, greyscale, bin, canny;
	cvtColor(image, hsv, CV_BGR2HSV);
	cvtColor(hsv, greyscale, CV_BGR2GRAY);
	threshold(greyscale, bin, 120, 255, THRESH_BINARY_INV | THRESH_OTSU);
	Canny(greyscale, canny, 50, 200, 3);

	//Houghlines
	vector<Vec2f> lines;
	/*HoughLines(canny, lines, 1, CV_PI / 180, 100, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(image, pt1, pt2, Scalar(0, 0, 0), 3, CV_AA);
	}
	*/
	//houghlinesp
	vector<Vec4i> lines2;
	Mat linesRes = Mat::zeros(image.size(), image.type());
	HoughLinesP(canny, lines2, 1, CV_PI / 90, 90, 70, 15);
	for (size_t i = 0; i < lines2.size(); i++)
	{
		Vec4i l = lines2[i];
		rectangle(linesRes, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0),1,8,0 );
		//line(test, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 5, CV_AA);
	}
	imshow("Houghlines", linesRes);
	imshow("Canny", canny);
}

static vector<Rect> floodFillPostprocess(Mat& img, int rows, int cols, const Scalar& colorDiff = Scalar::all(1))
{
	vector<Rect> rects;
	CV_Assert(!img.empty());
	RNG rng = theRNG();
	Mat mask(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)
			{
				Scalar newVal(rng(256), rng(256), rng(256));
				Rect r;
				floodFill(img, mask, Point(x, y), newVal, &r, colorDiff, colorDiff);
				if (r.width > 20 && r.height > 20 && r.width < cols && r.height <rows) {
					rects.push_back(r);
				}
			}
		}
	}
	return rects;
}

bool mergeRects(vector<Rect> rectangles, vector<Rect> &newRects) {
	//vector <Rect> newRects;
	vector <bool> used(rectangles.size());
	bool changes = false;
	std::fill(used.begin(), used.end(), false);
	for (int rectNo = 0; rectNo < (int)rectangles.size(); rectNo++) {
		Rect curRect = rectangles[rectNo];
		if (!used.at(rectNo)) {
			for (int rectNo2 = 0; rectNo2 < (int)rectangles.size(); rectNo2++) {
				if (rectNo != rectNo2) {
					Rect rect2 = rectangles[rectNo2];
					if (overlappingRects(curRect, rect2)) {
						//Update to larger rect
						curRect = getNewRect(curRect, rect2);
						used[rectNo2] = true;
						changes = true;
					}
				}
			}
			rectangles[rectNo] = curRect;
			//If valid rect add to vec
			if (curRect.width > 10 && curRect.height > 10) {
				newRects.push_back(curRect);
			}
		}
	}
	return changes;
}
void applyBoundingRect(Mat& image, vector<Rect> rectangles, Scalar colour) {
	//Iterate through all rectangles applying to image
	for (int rectNo = 0; rectNo < (int)rectangles.size(); rectNo++) {
		Rect boundRect = rectangles[rectNo];
		rectangle(image, boundRect.tl(), boundRect.br(), colour, 1, 8, 0);
	}
}
bool overlappingRects(Rect boundRect, Rect boundRect2) {
	return ((boundRect & boundRect2).area() > 0);
}
Rect getNewRect(Rect boundRect, Rect boundRect2) {
	int newX = min(boundRect.x, boundRect2.x);
	int newY = min(boundRect.y, boundRect2.y);
	int r1x2 = boundRect.x + boundRect.width;
	int r1y2 = boundRect.y + boundRect.height;
	int r2x2 = boundRect2.x + boundRect2.width;
	int r2y2 = boundRect2.y + boundRect2.height;
	int newWidth = max(r1x2, r2x2) - newX;
	int newHeight = max(r1y2, r2y2) - newY;
	return Rect(newX, newY, newWidth, newHeight);
}
static vector<Rect> filterBoundaryRects(vector<Rect> rectangles, int xmax, int ymax)
{
	vector<Rect> newRects;
	for (int rectNo = 0; rectNo < (int)rectangles.size(); rectNo++) {
		Rect boundRect = rectangles[rectNo];
		int x = boundRect.x;
		int y = boundRect.y;
			if (x != 0 && y != 0 && (y + boundRect.height < ymax) && (x + boundRect.width < xmax)) {
				newRects.push_back(boundRect);
			}
	}
	return newRects;
}
static vector<Rect> filterRects(vector<Rect> rectangles)
{
	vector<Rect> newRects;
	for (int rectNo = 0; rectNo < (int)rectangles.size(); rectNo++) {
		Rect boundRect = rectangles[rectNo];
		if (boundRect.width > 80 && boundRect.height > 80) {
				newRects.push_back(boundRect);
		}
	}
	return newRects;
}
static void getCroppedImages(Mat& image, vector<Rect> rects, int imageNo)
{
	for (int rectNo = 0; rectNo < (int)rects.size(); rectNo++) {
		Rect boundRect = rects[rectNo];
		Mat cropIm = image(boundRect);

		//Write cropped image to file 
		String fileLocation = "Paintings/" + to_string(imageNo) +"-"+ to_string(rectNo)+".jpg";
		imwrite(fileLocation, cropIm);
		imshow("cropped " + rectNo, cropIm);
		cvWaitKey();
		cvDestroyAllWindows();
	}
}

static vector<Rect> meanshiftApproach(Mat& image)
{
	Mat meanshiftImage;
	// (so-spatial rad, sr-colour window)
	pyrMeanShiftFiltering(image, meanshiftImage, 20, 56, 2);
	Mat meanshiftFlood = meanshiftImage.clone();
	vector<Rect> rects;
	rects = floodFillPostprocess(meanshiftFlood, meanshiftImage.rows, meanshiftImage.cols, Scalar::all(2));
	vector<Rect> newRects, mergedRects, filteredRects;
	//Filter out floor/ceiling segments
	newRects = filterBoundaryRects(rects, image.cols, image.rows);
	mergeRects(newRects, mergedRects);
	filteredRects = filterRects(mergedRects);
	return filteredRects;
}

//This is attempt to remove the frame from an image
//Returns an image which may have been cropped if it detected the frame
static Rect meanshiftApproach2(Mat& image)
{
	Mat meanshiftImage;
	// (so-spatial rad, sr-colour window)
	pyrMeanShiftFiltering(image, meanshiftImage, 18, 40, 2);
	Mat meanshiftFlood = meanshiftImage.clone();
	vector<Rect> rects;
	rects = floodFillPostprocess(meanshiftFlood, meanshiftImage.rows, meanshiftImage.cols, Scalar::all(2));
	vector<Rect> newRects, mergedRects, filteredRects;
	newRects = filterBoundaryRects(rects, image.cols, image.rows);
	mergeRects(newRects, mergedRects);
	//applyBoundingRect(image, mergedRects, Scalar(0, 0, 0xFF));
	
	//imshow("meanshift2AfterFlood", meanshiftFlood);
	//imshow("meanshift2", meanshiftImage);
	//waitKey();
	//destroyAllWindows();
	if (mergedRects.size() == 1) {
		Rect r = mergedRects.back();
		if (r.area() > (image.cols*image.rows*0.30)) { //Only return if rect covers at least 50 % of original image
			return r;
		}
	}
	return Rect(0,0,0,0);
}
static void applyCanny(Mat& image) {
	Mat greyscaleImage;
	cvtColor(image, greyscaleImage, CV_BGR2GRAY);
	Mat greyscaleCopy = greyscaleImage.clone();

	//Edge detection
	Mat edgesResult;
	Canny(greyscaleImage, edgesResult, 80, 180, 3);
	edgesResult.convertTo(edgesResult, CV_8U);
	imshow("greysscale", greyscaleCopy);
	imshow("canny", edgesResult);
	waitKey();
	destroyAllWindows();
}

//Computes correlation value from Hue & Sat histogram comparison 
//Between two images
static double compareImages(Mat& image1, Mat& image2)
{
	Mat im1, im2;
	cvtColor(image1, im1, COLOR_BGR2HSV);
	cvtColor(image2, im2, COLOR_BGR2HSV);

//	imshow("im1", im1);
	//imshow("im2", im2);

	//Setup params of HS histogram 
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };

	MatND im1Hist, im2Hist;

	calcHist(&im1, 1, channels, Mat(), im1Hist, 2, histSize, ranges, true, false);
	normalize(im1Hist, im1Hist, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&im2, 1, channels, Mat(), im2Hist, 2, histSize, ranges, true, false);
	normalize(im2Hist, im2Hist, 0, 1, NORM_MINMAX, -1, Mat());

	double corr = compareHist(im1Hist, im2Hist, CV_COMP_CORREL);
	
	Mat display_image2 = Mat::zeros(image2.size(), CV_8UC3);
	Draw1DHistogram(&im2Hist, 1, display_image2);
	imshow("im2 histogram", display_image2);

	Mat display_image = Mat::zeros(image1.size(), CV_8UC3);
	Draw1DHistogram(&im1Hist, 1, display_image);
	imshow("Im1 histogram", display_image);
	
	waitKey(); 
	destroyAllWindows();
	return corr;
}

static double templateMatching(Mat& im1, Mat& temp)
{
	Mat greyIm, greyTemp;
	cvtColor(im1, greyIm, COLOR_BGR2GRAY);
	cvtColor(temp, greyTemp, COLOR_BGR2GRAY);
	//imshow("greyIm", greyIm);
	//imshow("greyTemp", greyTemp);
	Mat result;
	int result_cols = im1.cols - temp.cols + 1;
	int result_rows = im1.rows - temp.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);
	matchTemplate(greyIm, greyTemp, result, CV_TM_CCOEFF_NORMED);
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	return maxVal;
}

void calcDICE(vector<vector<Point>> gt, vector<Rect> results) {
	vector<Rect> gtRects;
	double areaGT = 0.0;
	for (int i = 0; i < gt.size(); i++) {
		vector<Point> pts = gt.at(i);
	//	Rect r = Rect(pts.at(0).x, pts.at(0).y, pts.at(1).x, pts.at(1).y, pts.at(2).x, pts.at(2).y, pts.at(3).x, pts.at(3).y);
	//	areaGT = areaGT + r.area();
	//	gtRects.push_back(r);
	}
	//This shows the ground truth image
	applyBoundingRect(gtImage, gtRects, (0, 255, 0));
	imshow("GT", gtImage);

	double areaOverlap = 0.0;
	double areaRes = 0.0;
	for (int i = 0; i < results.size(); i++) {
		double maxOverlap = 0.0;
		Rect r1 = results.at(i);
		areaRes = areaRes + r1.area();
		for (int j = 0; j < gtRects.size(); j++) {
			Rect r2 = gtRects.at(j);
			double overlap = (r1 & r2).area();
			maxOverlap = max(maxOverlap, overlap);
		}
		areaOverlap = areaOverlap + maxOverlap;
	}
	double diceRes = (2 * areaOverlap) / (areaGT + areaRes);
	dice.push_back(diceRes);
}
static void displayGT(Mat& image, vector<vector<Point>> pts, int galleryNo)
{
	for (int i = 0; i < pts.size(); i++) {
		vector<Point> t = pts.at(i);
		polylines(image, t, false, Scalar(255, 0, 0));
	}
	//Write Ground truth image to file 
	String fileLocation = "GT/" + to_string(galleryNo)+ ".jpg";
	imwrite(fileLocation, image);
	imshow("GT", image);
}

int main(int argc, const char** argv)
{
	char* file_location = "Paintings/";
	char* image_files[] = {

		"Gallery1.jpg", //0
		"Gallery2.jpg",//1
		"Gallery3.jpg", //2
		"Gallery4.jpg",	//3	
		"Painting1.jpg", //4
		"Painting2.jpg",//5
		"Painting3.jpg",//6
		"Painting4.jpg",//7
		"Painting5.jpg",//8
		"Painting6.jpg",//9
		//"3-1Test.jpg",
		//"Painting6Test.jpg",
		//"Painting3Test.jpg",
		"0-0.jpg", //10
		"0-1.jpg", //11
		"1-0.jpg", //12
		"1-1.jpg",//13
		"2-0.jpg", //14
		"2-1.jpg", //15
		"2-2.jpg", //16
		"3-0.jpg", //17
		"3-1.jpg", //18
		"3-2.jpg", //19
		"3-3.jpg" //20
		//"4_1_seg.jpg",//19	
		//"Painting5_seg.jpg"//20
	};

	// Load images
	int number_of_images = sizeof(image_files) / sizeof(image_files[0]);
	Mat* image = new Mat[number_of_images];
	for (int file_no = 0; (file_no < number_of_images); file_no++)
	{
		string filename(file_location);
		filename.append(image_files[file_no]);
		image[file_no] = imread(filename, -1);
		if (image[file_no].empty())
		{
			cout << "Could not open " << image[file_no] << endl;
			return -12;
		}
	}
	Mat* gallerys = new Mat[NO_GALLERYS];
	Mat* templates = new Mat[NO_PAINTINGS];
	int noCropped = number_of_images - (NO_GALLERYS + NO_PAINTINGS);
	Mat* cropped = new Mat[noCropped];

	for (int i = 0; i < number_of_images; i++) {
		if (i < 4) {
			gallerys[i] = image[i];
		}
		else if (i > 9) {
			cropped[i-10] = image[i];
		}
		else {
			templates[i - 4] = image[i];
		}
	}

	//Ground truth for images (Gallery 1-4)
	groundTruths.push_back({ { Point(212, 261) , Point(445, 225), Point(428, 725) , Point(198,673) },{ Point(686, 377) , Point(1050, 361), Point(1048, 705) , Point(686,652) } });
	groundTruths.push_back({ { Point(252, 279) , Point(691, 336), Point(695, 662) , Point(258,758) },{ Point(897, 173) , Point(1063, 234), Point(1079, 672) , Point(917,739) },{ Point(1174, 388) , Point(1221, 395), Point(1216, 544) , Point(1168,555) } });
	groundTruths.push_back({ { Point(68, 329) , Point(350, 337), Point(351, 545) , Point(75,558) },{ Point(629, 346) , Point(877, 350), Point(873, 517) , Point(627,530) },{ Point(1057, 370) , Point(1187, 374), Point(1182, 487) , Point(1053,493) } });
	groundTruths.push_back({ { Point(176,348) , Point(298,347), Point(307,481) , Point(184,475) },{ Point(469,343) , Point(690,338), Point(692,495) , Point(472,487) },{ Point(924, 349) , Point(1161,344), Point(1156, 495) , Point(924,488) } });

	groundTruthStrings.push_back({ "Painting 2", "Painting1" });
	groundTruthStrings.push_back({ "Painting 3", "Painting2", "Painting1" });
	groundTruthStrings.push_back({ "Painting 4", "Painting5", "Painting6" });
	groundTruthStrings.push_back({ "Painting 4", "Painting5", "Painting6" });




	for (int galleryNo = 1; galleryNo < NO_GALLERYS; galleryNo++) {
		Mat currentImage = gallerys[galleryNo];
		Mat imCopy = currentImage.clone();
		vector<Rect> gallerySegments = meanshiftApproach(imCopy);
		vector<Rect> resRect;
		vector<int> resPaintings;
		cout << "CroppedNo " << "Template No  " << " | " << "CorrRes" << " tempmatchRes\n";
		for (int galleySegNo = 0; galleySegNo < gallerySegments.size(); galleySegNo++) {
			Rect r = gallerySegments.at(galleySegNo);

			//Meanshift to try remove frame if poss
			Rect paintingRect = meanshiftApproach2(imCopy(r));
			if (paintingRect.area() == 0) {
				paintingRect = r;
			}
			else {
				paintingRect.x += r.x;
				paintingRect.y += r.y;
			}
			Mat croppedIm =  imCopy(paintingRect);
			imshow("croppedIm", croppedIm);
			waitKey();
			destroyAllWindows();
			//imshow("Im", imCopy);
			double tempMax = 0.0;
			int maxIndex = 0;
			double histRes = 0.0;
			for (int templateNo = 0; templateNo < NO_PAINTINGS; templateNo++)
			{
				Mat template1;
				//Resize template painting to same as croppedIm
				resize(templates[templateNo], template1, croppedIm.size());
				imshow("template", template1);
				waitKey();
				destroyAllWindows();
				//Compare the hist of both
				double corr = compareImages(croppedIm, template1);

				//Template Matching Res
				double tempRes = templateMatching(croppedIm, template1);
				if (tempRes > tempMax) {
					tempMax = tempRes;
					maxIndex = templateNo;
					histRes = corr;
				}
				//print result
				cout << galleySegNo << " , " << templateNo + 1 << " | " << corr << " | " << tempRes << "\n";
				//	waitKey();
					//destroyAllWindows();
			}
			double threshold = tempMax + histRes;
			if (threshold >= 0.59) {
				cout << "Recognised as Painting" << maxIndex + 1 << "   The thres: " << threshold ;
				resRect.push_back(paintingRect);
				resPaintings.push_back(maxIndex + 1);
				cout <<"\n" << paintingRect.x << " " << paintingRect.y << " - " << paintingRect.x + paintingRect.width << " " << paintingRect.y + paintingRect.height;
			}
			else {
				cout << "Failed to recognise this segment";
			}
			cout << "\n\n";
		}
		applyBoundingRect(currentImage, resRect, Scalar(255, 0, 0));
		for (int i = 0; i < resRect.size(); i++) {
			String text = "Painting " + to_string(resPaintings.at(i));
			putText(currentImage, text, Point(resRect.at(i).x+5, resRect.at(i).y-20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 235,0),2, false);
		}
		displayGT(imCopy, groundTruths.at(galleryNo), galleryNo);
		imshow("res", currentImage);
		//Write resulting image to file 
		String fileLocation = "Results/" + to_string(galleryNo) + ".jpg";
		imwrite(fileLocation, currentImage);
		//waitKey();
		//destroyAllWindows();
	}
	waitKey();
}
static void histApproach(Mat& image)
{
	hist(image);
	ComputeHistogram(image);
}

//From: https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/back_projection/back_projection.html
/*static void Hist_and_Backproj(Mat& src,int, void*)
{
Mat hsv, hue;
cvtColor(src, hsv, CV_BGR2HSV);
/// Use only the Hue value
hue.create(hsv.size(), hsv.depth());
int ch[] = { 0, 0 };
mixChannels(&hsv, 1, &hue, 1, ch, 1);
MatND hist;
int bins = 25;
Mat hue;
int histSize = 25;
float hue_range[] = { 0, 180 };
const float* ranges = { hue_range };

/// Get the Histogram and normalize it
calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

/// Get Backprojection
MatND backproj;
calcBackProject(&hue, 1, 0, hist, backproj, &ranges, 1, true);

/// Draw the backproj
imshow("BackProj", backproj);

/// Draw the histogram
int w = 400; int h = 400;
int bin_w = cvRound((double)w / histSize);
Mat histImg = Mat::zeros(w, h, CV_8UC3);

for (int i = 0; i < bins; i++)
{
rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)), Scalar(0, 0, 255), -1);
}

imshow("Histogram", histImg);
}
*/