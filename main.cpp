#include "Utilities.h"
#include <iostream>
#include <fstream>
#define NO_GALLERYS 4
#define NO_PAINTINGS 6

using namespace std;
vector<vector<vector<Point>>> groundTruths;
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
	Rect r1 = boundRect;
	//Add to r1 width and height to allow it to overlap with close by rects
	r1.width += 0;
	r1.height += 0;
	return ((r1 & boundRect2).area() > 0);
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
// r c
// Rect(c/3,r/3,c/3,r/3)
static void meanshiftApproach(Mat& image, int imageNo)
{
	//cols*rows// image 2 = 1296*968
	Mat greyscaleImage, meanshiftImage, meanshiftImage2;
	// (so-spatial rad, sr-colour window)
	pyrMeanShiftFiltering(image, meanshiftImage, 20, 56, 2);
	//	cvtColor(meanshiftImage, greyscaleImage, CV_BGR2GRAY);
	//Mat test = greyscaleImage.clone();
	Mat meanshiftFlood = meanshiftImage.clone();
	//floodFillPostprocess(test, Scalar::all(2));
	vector<Rect> rects;
	rects = floodFillPostprocess(meanshiftFlood, meanshiftImage.rows, meanshiftImage.cols, Scalar::all(2));
	vector<Rect> newRects, mergedRects, filteredRects;

	//Filter out floor/ceiling segments
	newRects = filterBoundaryRects(rects, image.cols, image.rows);
	//newRects = rects;
	mergeRects(newRects, mergedRects);
	filteredRects = filterRects(mergedRects);
	//applyBoundingRect(image, filteredRects, (0, 0, 0xFF));
	getCroppedImages(image, filteredRects, imageNo);
	imshow("meanshift22AfterFlood", meanshiftFlood);
	imshow("meanshift22", meanshiftImage);
	namedWindow("rects", WINDOW_NORMAL);
	imshow("rects", image);
	//imshow("bin", bin);
}


static void meanshiftApproach2(Mat& image)
{
	//cols*rows// image 2 = 1296*968
	Mat greyscaleImage, meanshiftImage, meanshiftImage2;
	// (so-spatial rad, sr-colour window)
	pyrMeanShiftFiltering(image, meanshiftImage, 18, 38, 2);
	//	cvtColor(meanshiftImage, greyscaleImage, CV_BGR2GRAY);
	//Mat test = greyscaleImage.clone();
	Mat meanshiftFlood = meanshiftImage.clone();
	//floodFillPostprocess(test, Scalar::all(2));
	vector<Rect> rects;
	rects = floodFillPostprocess(meanshiftFlood, meanshiftImage.rows, meanshiftImage.cols, Scalar::all(2));
	vector<Rect> newRects, mergedRects, filteredRects;
	newRects = filterBoundaryRects(rects, image.cols, image.rows);
	mergeRects(newRects, mergedRects);
	applyBoundingRect(image, mergedRects, (0, 0, 0xFF));

	imshow("meanshift22AfterFlood", meanshiftFlood);
	imshow("meanshift22", meanshiftImage);
	namedWindow("rects", WINDOW_NORMAL);
	imshow("rects", image);
	//imshow("bin", bin);
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
	Mat* cropped = new Mat[number_of_images - (NO_GALLERYS + NO_PAINTINGS)];
	for (int i = 0; i < number_of_images; i++) {
		if (i < 4) {
			gallerys[i] = image[i];
		}
		else if (i > 9) {
			cropped[i-10] = image[i];
			meanshiftApproach2(image[i]);
			waitKey();
			destroyAllWindows();
		}
		else {
			templates[i - 4] = image[i];
		}
	}
	int choice;
	int i = 0;
	int noTemplates = 6;
/*	meanshiftApproach2(cropped[0]);
	waitKey();
	Mat croppedIm = cropped[2];
	MatND imHist = ComputeHistogram(croppedIm);
	waitKey();
	*/
	//Mat croppedIm = templates[0];
	/*for (int templateNo = 0; templateNo < noTemplates; templateNo++)
	{

		
		Mat newTemplate, greyTemplate, greyIm;
		Mat bin, c1, c2;
		Size s = Size(200,200);
		resize(templates[templateNo], newTemplate, croppedIm.size());
		newTemplate = cropped[3];
		MatND templateHist = ComputeHistogram(newTemplate);
		waitKey();
		//hist(croppedIm);
		//hist(newTemplate);
		double corr = compareHist(templateHist, imHist, CV_COMP_CORREL);
		
		cvtColor(newTemplate, c1, COLOR_BGR2HSV);
		cvtColor(croppedIm, c2, COLOR_BGR2HSV);
		cvtColor(c1, greyTemplate, COLOR_BGR2GRAY);
		cvtColor(c2, greyIm, COLOR_BGR2GRAY);
		
		//threshold(greyIm, bin,150, 255, CV_THRESH_BINARY);

		//cvtColor(newTemplate, greyTemplate, COLOR_BGR2HSV);
		//cvtColor(croppedIm, greyIm, COLOR_BGR2HSV);
		imshow("resizedTemplate", greyTemplate);
		imshow("greyIm", greyIm);
		waitKey();
		destroyAllWindows();
		Mat result;
		int result_cols = croppedIm.cols - newTemplate.cols + 1;
		int result_rows = croppedIm.rows - newTemplate.rows + 1;
		result.create(result_rows, result_cols, CV_32FC1);

		matchTemplate(greyTemplate, greyIm, result, TM_CCOEFF_NORMED);

		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		cout << "result" << templateNo << " : " << maxVal << " | " << corr << "\n";
	}*/
	
	
	//The following goes from original input gallery to individual regions in galleries
	//These are cropped and saved for testing purposes

	/*while (i < NO_GALLERYS) {
		Mat currentImage = gallerys[i];
		Mat imCopy = currentImage.clone();
		//Mat k = kmeans_clustering(imCopy, 5, 1);
		//houghLinesApproach(k);
		//histApproach(k);
		meanshiftApproach(imCopy,i);
		//imshow("imagek", k);
		i++;
		choice = cvWaitKey();
		cvDestroyAllWindows();
	}*/

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