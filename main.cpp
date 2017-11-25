#include "Utilities.h"
#include <iostream>
#include <fstream>
#define NO_GALLERYS 4
#define NO_PAINTINGS 6

using namespace std;
vector<vector<vector<Point>>> groundTruths;
vector<double> dice;
Mat gtImage;

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
	int choice;
	int i = 0;
	while (i < NO_GALLERYS){
		Mat k = kmeans_clustering(gallerys[i], 3, 1);
		Mat greyscale,bin;
		cvtColor(k, greyscale, CV_BGR2GRAY);
		threshold(greyscale, bin, 120, 255, THRESH_BINARY | THRESH_OTSU);
		imshow("Image "+i,gallerys[i]);
		imshow("gret " + i, greyscale);
		imshow("kmeans(2,2) " + i, k);
		imshow("bin " + i, bin);
		i++;
		choice = cvWaitKey();
		cvDestroyAllWindows();
	} 
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

