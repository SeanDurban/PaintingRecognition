#include "Utilities.h"
#include <iostream>
#include <fstream>
#define NO_GALLERYS 4
#define NO_PAINTINGS 6
#define MIN_OVERLAP_THRESHOLD 0.6
#define NO_PAINTINGS_IN_GALLERIES 11

using namespace std;
vector<vector<vector<Point>>> groundTruths;
vector<vector<String>> groundTruthStrings;
vector<vector<int>> groundTruthPaintings;
vector<double> dice;
double precisionSum;
double recallSum;
double accuracySum;
double f1Sum;

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

//Applies floodfill to a meanshift filtered image
//Segementation step
//Returns a vector of bounding rects for the segments found
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
//Filters out rects which are on the boundary of the image (i.e include points at the edge)
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
//Filters out rects below a set size
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
	}
}

//Returns a vector of rects which represent the segments believed to be paintings in the image provided
//Uses meanshift segmentation and rect classification to determine this vector.
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
//Returns a rect of the image excluding the frame if one found
//Otherwise returns empty rect
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
	if (mergedRects.size() == 1) {
		Rect r = mergedRects.back();
		if (r.area() > (image.cols*image.rows*0.30)) { //Only return if rect covers at least 50 % of original image
			return r;
		}
	}
	return Rect(0,0,0,0);
}

//Computes correlation value from Hue & Sat histogram comparison 
//Between two images
static double compareImages(Mat& image1, Mat& image2)
{
	Mat im1, im2;
	cvtColor(image1, im1, COLOR_BGR2HSV);
	cvtColor(image2, im2, COLOR_BGR2HSV);


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
	
	/* This code displays the generated histograms
	Mat display_image2 = Mat::zeros(image2.size(), CV_8UC3);
	Draw1DHistogram(&im2Hist, 1, display_image2);
	imshow("im2 histogram", display_image2);

	Mat display_image = Mat::zeros(image1.size(), CV_8UC3);
	Draw1DHistogram(&im1Hist, 1, display_image);
	imshow("Im1 histogram", display_image);
	
	waitKey(); 
	destroyAllWindows();
	*/
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

static void displayGT(Mat& image, vector<vector<Point>> pts, int galleryNo, vector<String> gtPaintings)
{
	for (int i = 0; i < pts.size(); i++) {
		String text = gtPaintings.at(i);
		vector<Point> t = pts.at(i);
		polylines(image, t, false, Scalar(255, 0, 0));
		putText(image, text, Point(pts.at(i).at(0).x + 10, pts.at(i).at(0).y + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 235, 0), 2, false);
	}
	//Write Ground truth image to file 
	String fileLocation = "GT/" + to_string(galleryNo)+ ".jpg";
	imwrite(fileLocation, image);
	imshow("GT", image);
}
static Rect zoom(Mat& image, double zoomFactor)
{
	int x = image.cols*zoomFactor;
	int y = image.rows*zoomFactor;
	int w = (image.cols *(1 - zoomFactor)) - x;
	int h = (image.rows*(1 - zoomFactor)) - y;
	return Rect(x, y, w, h);
}
static bool isCorrectPrediction(vector<Point> groundTruth, Rect result)
{
	int x = groundTruth.at(0).x; int y = groundTruth.at(0).y;
	int w = groundTruth.at(2).x - x; int h = groundTruth.at(2).y - y;
	Rect gtRect = Rect(x, y, w, h);
	double minAreaThreshold = gtRect.area() * MIN_OVERLAP_THRESHOLD;
	double area =(gtRect & result).area();
	return area > minAreaThreshold;
}

static void calculateMetrics(int tp, int fp,int tn, int fn)
{
	double accuracy = (double) (tp + tn) / (tp + fp + fn + tn);
	double precision = (double) (tp) / (tp + fp);
	double recall = (double) (tp) / (tp + fn);
	double f1Score = (2 * (recall*precision)) / (recall + precision);
	accuracySum += accuracy; precisionSum += precision; recallSum += recall; f1Sum += f1Score;
	cout << "TP  FP  TN  FN  |  Precision   |  Recall  |  Accuracy  |  F1 Score\n";
	cout << tp <<"  "<< fp << "  " << tn << "  " << fn << "  |  " << precision << "  |  " << recall << "  |  " << accuracy << "  |  " << f1Score << "\n";
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
		"Painting6.jpg"//9
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

	for (int i = 0; i < number_of_images; i++) {
		if (i < 4) {
			gallerys[i] = image[i];
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

	groundTruthStrings.push_back({ "Painting 2", "Painting 1" });
	groundTruthStrings.push_back({ "Painting 3", "Painting 2", "Painting 1" });
	groundTruthStrings.push_back({ "Painting 4", "Painting 5", "Painting 6" });
	groundTruthStrings.push_back({ "Painting 4", "Painting 5", "Painting 6" });


	groundTruthPaintings.push_back({ 2,1});
	groundTruthPaintings.push_back({ 3,2,1 });
	groundTruthPaintings.push_back({ 4,5,6 });
	groundTruthPaintings.push_back({ 4,5,6 });
	
	precisionSum = 0.0;
	recallSum = 0.0;
	accuracySum = 0.0;
	f1Sum = 0.0;

	for (int galleryNo = 0; galleryNo < NO_GALLERYS; galleryNo++) {
		cout << "\nGallery No : " << galleryNo+1 << "\n";
		Mat currentImage = gallerys[galleryNo];
		Mat imCopy = currentImage.clone();
		vector<Rect> gallerySegments = meanshiftApproach(imCopy);
		vector<Rect> resRect;
		vector<int> resPaintings;
		vector<int> gtPaintings = groundTruthPaintings.at(galleryNo);

		for (int galleySegNo = 0; galleySegNo < gallerySegments.size(); galleySegNo++) {
			Rect r = gallerySegments.at(galleySegNo);

			//Meanshift to try remove frame if poss
		/*	Rect paintingRect = meanshiftApproach2(imCopy(r));
			if (paintingRect.area() == 0) {
				paintingRect = r;
			}
			else {
				paintingRect.x += r.x;
				paintingRect.y += r.y;
			}
			Mat croppedIm =  imCopy(paintingRect);*/

			Mat croppedIm1 = imCopy(r);
			Rect paintingRect= zoom(imCopy(r),0.14);
			Mat croppedIm = croppedIm1(paintingRect);
			paintingRect.x += r.x;
			paintingRect.y += r.y;

			double tempMax = 0.0;
			int maxIndex = 0;
			double histRes = 0.0;
			for (int templateNo = 0; templateNo < NO_PAINTINGS; templateNo++)
			{
				Mat templateResized;
				//Resize template painting to same as croppedIm
				resize(templates[templateNo], templateResized, croppedIm.size());

				//Compare the hist of both
				double corr = compareImages(croppedIm, templateResized);

				//Template Matching Res
				double tempRes = templateMatching(croppedIm, templateResized);
				if (tempRes > tempMax) {
					tempMax = tempRes;
					maxIndex = templateNo;
					histRes = corr;
				}
			}
			double threshold = (tempMax + histRes)/2;
			if (threshold >= 0.4) {
				cout << "Recognised Painting" << maxIndex + 1 <<"\n";
				resRect.push_back(paintingRect);
				resPaintings.push_back(maxIndex + 1);
			}

		}
		int tp = 0; int fp = 0; int tn = 0; int fn = 0;
		applyBoundingRect(currentImage, resRect, Scalar(255, 0, 0));
		for (int i = 0; i < resRect.size(); i++) {
			int index = find(gtPaintings.begin(), gtPaintings.end(), resPaintings.at(i)) - gtPaintings.begin();
			if (index >= 0 && index < gtPaintings.size()) {
				if (isCorrectPrediction(groundTruths.at(galleryNo).at(index), resRect.at(i))){
					tp++;
				}
				else {
					fp++;
				}
			}
			String text = "Painting " + to_string(resPaintings.at(i));
			putText(currentImage, text, Point(resRect.at(i).x+10, resRect.at(i).y+20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 235,0),2, false);
		}
		fn += gtPaintings.size() - tp -fp;
		tn = 0; 
		calculateMetrics(tp, fp, tn, fn);
		
		//Display both result and ground truth
		displayGT(imCopy, groundTruths.at(galleryNo), galleryNo, groundTruthStrings.at(galleryNo));
		imshow("res", currentImage);
	}
	double avgPrecision = precisionSum / NO_GALLERYS;
	double avgRecall = recallSum / NO_GALLERYS;
	double avgAccuracy = accuracySum / NO_GALLERYS;
	double avgF1 = f1Sum / NO_GALLERYS;
	cout << "\n\nAverage over all Gallerys\n";
	cout << "Precision | Recall | Accuracy | F1 Score\n";
	cout << avgPrecision << "  |  " << avgRecall << "  |  " << avgAccuracy << "  |  " << avgF1 << "\n";
	waitKey();
}
