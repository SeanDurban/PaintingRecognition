#include "Utilities.h"
#include <iostream>
#include <fstream>
using namespace std;
vector<vector<vector<Point>>> groundTruths;
vector<double> dice;
Mat gtImage;
const int MIN_CHILD_CONTOURS = 20;
const int MAX_CONTOUR_AREA = (250 * 250);
const int MIN_CONTOUR_AREA = 50;
const int MIN_SIGN_AREA = 250;
int main(int argc, const char** argv)
{
	char* file_location = "Notices/";
	char* image_files[] = {

		"Notice1.jpg", //0
		"Notice2.jpg",//1
		"Notice3.jpg", //2
		"Notice4.jpg",	//3	
		"Notice5.jpg", //4
		"Notice6.jpg",//5
		"Notice7.jpg",//6
		"Notice8.jpg"//7
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
			return -1;
		}
	}
	//Ground truth for images
	groundTruths.push_back({ { Point(34, 17) , Point(286, 107) },{ Point(32, 117) ,Point(297, 223) },{ Point(76, 234) , Point(105, 252) } });
	groundTruths.push_back({ { Point(47, 191) ,Point(224, 253) } });
	groundTruths.push_back({ { Point(142, 121) , Point(566, 392) } });
	groundTruths.push_back({ { Point(157,72) , Point(378, 134) },{ Point(392, 89) , Point(448, 132) },{ Point(405, 138) , Point(442, 152) },{ Point(80, 157) ,Point(410, 245) },{ Point(82, 258) , Point(372, 322) } });
	groundTruths.push_back({ { Point(112, 73) , Point(598, 170) },{ Point(108, 178) ,Point(549, 256) },{ Point(107, 264) ,Point(522, 352) } });
	groundTruths.push_back({ { Point(91, 54) , Point(446, 227) } });
	groundTruths.push_back({ { Point(64, 64) , Point(476, 268) },{ Point(529, 126) , Point(611, 188) },{ Point(545, 192) , Point(603, 211) },{ Point(210, 305) , Point(595, 384) } });
	groundTruths.push_back({ { Point(158, 90) , Point(768, 161) },{ Point(114, 174) , Point(800, 279) } });
	
	int choice;
	int i = 0;
	do
	{
		choice = cvWaitKey();
		cvDestroyAllWindows();
		gtImage = image[i].clone();
		//Get rectangles encasing text in image[i]
		vector<Rect> resultRects = getTextRects(image[i]);
		//Calculate DICE coefficient
		calcDICE(groundTruths.at(i), resultRects);
		cout << "Dice for " << i+1 << " : " << dice.at(i) << "\n";
		i++;

	} while (i < number_of_images);
}
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
void calcDICE(vector<vector<Point>> gt, vector<Rect> results) {
	vector<Rect> gtRects;
	double areaGT = 0.0;
	for (int i = 0; i < gt.size(); i++) {
		vector<Point> pts = gt.at(i);
		int x = pts.at(0).x;
		int y = pts.at(0).y;
		int w = pts.at(1).x - x;
		int h = pts.at(1).y - y;
		Rect r = Rect(x, y, w, h);
		areaGT = areaGT + r.area();
		gtRects.push_back(r);
	}
	//This shows the ground truth image
	applyBoundingRect(gtImage, gtRects, (0, 255, 0));
	imshow("GT",gtImage);

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
vector<Rect> getTextRects(Mat& image) {
	//Meanshift and convert to greyscale
	Mat greyscaleImage, meanshiftImage;
	pyrMeanShiftFiltering(image, meanshiftImage, 45, 20, 2);
	cvtColor(meanshiftImage, greyscaleImage, CV_BGR2GRAY);
	Mat greyscaleCopy = greyscaleImage.clone();

	//Edge detection
	Mat edgesResult;
	Canny(greyscaleImage, edgesResult, 80, 180, 3);
	edgesResult.convertTo(edgesResult, CV_8U);
	Mat edgesResultCopy = edgesResult.clone();

	//Closing (dilate and erode) on ED result
	Mat closeRes = edgesResult.clone();
	morphologyEx(closeRes, closeRes, MORPH_CLOSE, Mat(), Point(-1, -1), 2);

	//Connected components on the closing result
	vector<vector<Point>> contoursFound;
	vector<Vec4i> hierarchy;
	findContours(closeRes, contoursFound, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	//Classify edges to get sign
	vector<Point> signContour;
	signContour = contoursFound[0];
	int curMaxArea = 0;
	for (int contourNum = 0; (contourNum < (int)contoursFound.size()); contourNum++)
	{
		vector<Point> curContour = contoursFound[contourNum];
		//Check is suitable size
		if (contourArea(contoursFound[contourNum]) > MIN_SIGN_AREA) {
			//Check if is parent
			int child = hierarchy[contourNum].val[2];
			if (child != -1) {
				int count = 1;
				while (hierarchy[child].val[0] != -1) {
					//Check if valid contour
					int a = contourArea(contoursFound[child]);
					if (a > MIN_CONTOUR_AREA && a< MAX_CONTOUR_AREA) {
						int child2 = hierarchy[child].val[2];
						if (child2 != -1) {
							count++;
							while (hierarchy[child2].val[0] != -1) {
								count++;
								child2 = hierarchy[child2].val[0];
							}
						}
						count++;
					}
					//Go to next contour
					child = hierarchy[child].val[0];
				}
				if (count >= MIN_CHILD_CONTOURS) {
					if (contourArea(contoursFound[contourNum]) > curMaxArea) {
						signContour = contoursFound[contourNum];
						curMaxArea = contourArea(contoursFound[contourNum]);
					}
				}
			}
		}
	}

	//Get rectangle around sign contour and crop to image to sign size
	Rect signRect = boundingRect(signContour);
	Mat cropped = image(signRect);
	//Apply k means clustering with k =5
	Mat clustered_image;
	clustered_image = kmeans_clustering(cropped, 2, 5);
	//Convert k means result to greyscale and apply threshold
	cvtColor(clustered_image, clustered_image, CV_BGR2GRAY);
	threshold(clustered_image, clustered_image, 150, 255, THRESH_BINARY_INV);
	Mat clusteredCopy = clustered_image.clone();

	vector<vector<Point>> contoursFound2;
	//Connected components on the ED
	findContours(clusteredCopy, contoursFound2, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	vector<Rect> rectangles;
	for (int contour_number = 0; (contour_number<(int)contoursFound2.size()); contour_number++)
	{
		Rect boundRect = boundingRect(Mat(contoursFound2[contour_number]));
		//If valid rect
		if (boundRect.width <150 && boundRect.height < 150 && boundRect.height> 5 && boundRect.width>5) {
			rectangles.push_back(boundRect);
		}
	}

	bool changes = true;
	vector<Rect> newRects;
	vector<Rect> rectsToMerge = rectangles;
	//Merge the rectangles until no more changes (rects merged) on last iteration
	while (changes) {
		newRects.clear();
		changes = mergeRects(rectsToMerge, newRects);
		rectsToMerge = newRects;
	}

	//Apply rects to cropped imageS
	applyBoundingRect(cropped, newRects, (0, 0, 0xFF));

	//Outputs
	imshow("image", image);
	return newRects;
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
		rectangle(image, boundRect.tl(), boundRect.br(), colour, 2, 8, 0);
	}
}
bool overlappingRects(Rect boundRect, Rect boundRect2) {
	Rect r1 = boundRect;
	//Add to r1 width and height to allow it to overlap with close by rects
	r1.width += 25;
	r1.height += 8;
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

