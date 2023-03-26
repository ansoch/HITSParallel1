//#pragma intrinsic

#include <iostream>
#include <vector>
#include "lodepng.h"
#include "lodepng.cpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <ctime> 
#include <emmintrin.h>
#include <immintrin.h>

using namespace std;
struct pixel {
	unsigned char R, G, B;
};

void encodeOneStep(const char* filename, vector<unsigned char>& image, unsigned width, unsigned height) {
	//Encode the image
	unsigned error = lodepng::encode(filename, image, width, height);

	//if there's an error, display it
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

vector<unsigned char> NegativeFilterСoherent(vector<unsigned char> image) {
	int i = 0;
	for (auto& e : image) {
		i++;
		if (i != 4) e = 255 - e;
		else i = 0;
	}
	return image;
}

vector<unsigned char> MedianFilterCoherent(vector<unsigned char> image, unsigned width, unsigned height) {
	vector<vector<pixel>> pixelMatrix(height, vector<pixel>(width, pixel()));
	for (int i = 0; i < width * height * 4; i += 4) {
		pixelMatrix[i / (4 * height)][(i % (width * 4)) / 4].R = image[i];
		pixelMatrix[i / (4 * height)][(i % (width * 4)) / 4].G = image[i + 1];
		pixelMatrix[i / (4 * height)][(i % (width * 4)) / 4].B = image[i + 2];
	}

	int yAxis;
	int xAxis;

	vector<int> matrixR(255);
	vector<int> matrixG(255);
	vector<int> matrixB(255);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			yAxis = y < 7 ? 7 : y > height - 8 ? height - 8 : y;
			xAxis = x < 7 ? 7 : x > width - 8 ? width - 8 : x;
			yAxis -= 7; xAxis -= 7;
			for (int i = 0; i < 15; i++) {
				for (int j = 0; j < 15; j++) {
					matrixR[i * 15 + j] = pixelMatrix[yAxis + i][xAxis + j].R;
					matrixG[i * 15 + j] = pixelMatrix[yAxis + i][xAxis + j].G;
					matrixB[i * 15 + j] = pixelMatrix[yAxis + i][xAxis + j].B;
				}
			}

			nth_element(matrixR.begin(), matrixR.begin() + 225 / 2, matrixR.end());
			nth_element(matrixG.begin(), matrixG.begin() + 225 / 2, matrixG.end());
			nth_element(matrixB.begin(), matrixB.begin() + 225 / 2, matrixB.end());

			image[y * height * 4 + x * 4] = matrixR[225 / 2];
			image[y * height * 4 + x * 4 + 1] = matrixG[225 / 2];
			image[y * height * 4 + x * 4 + 2] = matrixB[225 / 2];
		}
	}
	return image;
}

vector<unsigned char> NegativeFilterOMP(vector<unsigned char> image) {
   #pragma omp parallel for shared(image)
	for (int i = 0; i < image.size(); i++) {
		if ((i + 1) % 4 != 0) image[i] = 255 - image[i];
	}
	return image;
}

vector<unsigned char> MedianFilterOMP(vector<unsigned char> image, unsigned width, unsigned height) {
	vector<vector<vector<int>>> pixelMatrix(height, vector<vector<int>>(width, vector<int>(3)));

	//распараллеливание не имеет смысла (без него быстрее)
    #pragma omp parallel for shared(pixelMatrix, image, height, width)
	for (int i = 0; i < width * height * 4; i += 4)
		for (int j = 0; j < 3; j++)
			pixelMatrix[i / (4 * height)][(i % (width * 4)) / 4][j] = image[i + j];

	#pragma omp barrier

	int yAxis;
	int xAxis;

    #pragma omp parallel for shared(image, width, height, pixelMatrix) private( yAxis, xAxis)
	for (int y = 0; y < height; y++) {

		for (int x = 0; x < width; x++) {
			vector<int> matrixR(255);
			vector<int> matrixG(255);
			vector<int> matrixB(255);

			yAxis = y < 7 ? 7 : y > height - 8 ? height - 8 : y;
			xAxis = x < 7 ? 7 : x > width - 8 ? width - 8 : x;
			yAxis -= 7; xAxis -= 7;

			for (int i = 0; i < 15; i++) {
				for (int j = 0; j < 15; j++) {
					matrixR[i * 15 + j] = pixelMatrix[yAxis + i][xAxis + j][0];
					matrixG[i * 15 + j] = pixelMatrix[yAxis + i][xAxis + j][1];
					matrixB[i * 15 + j] = pixelMatrix[yAxis + i][xAxis + j][2];
				}
			}

			nth_element(matrixR.begin(), matrixR.begin() + 225 / 2, matrixR.end());
			nth_element(matrixG.begin(), matrixG.begin() + 225 / 2, matrixG.end());
			nth_element(matrixB.begin(), matrixB.begin() + 225 / 2, matrixB.end());

			image[y * height * 4 + x * 4] = matrixR[225 / 2];
			image[y * height * 4 + x * 4 + 1] = matrixG[225 / 2];
			image[y * height * 4 + x * 4 + 2] = matrixB[225 / 2];
		}
	}
	return image;
}

vector<unsigned char> NegativeFilterVEC(vector<unsigned char> image) {
	__m128i invert = _mm_set1_epi8(255);
	//__m128i invert = _mm_set_epi8(255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0);
	//__m128i sum254 = _mm_set_epi8(0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255);

	for (int i = 0; i < image.size(); i += 16) {
		__m128i pixel = _mm_loadu_si128((__m128i*) & image[i]);

		pixel = _mm_sub_epi8(invert, pixel);
		pixel = _mm_insert_epi8(pixel, 255, 3); pixel = _mm_insert_epi8(pixel, 255, 7); pixel = _mm_insert_epi8(pixel, 255, 11); pixel = _mm_insert_epi8(pixel, 255, 15); //не убавляет времени

		//pixel = _mm_add_epi8(pixel, sum254);

		_mm_storeu_si128((__m128i*) & image[i], pixel);
		//image[i + 3] = 255; image[i + 7] = 255; image[i + 11] = 255; image[i+15] = 255;
	}
	return image;
}

vector<unsigned char> MedianFilterVEC(vector<unsigned char> image, unsigned width, unsigned height) {
	vector<unsigned char> pixelMatrixR(height * width);
	vector<unsigned char> pixelMatrixG(height * width);
	vector<unsigned char> pixelMatrixB(height * width);
	int index = 0;
	__m128i mat;
	for (int i = 0; i < height * width * 4; i+=16) {
		mat = _mm_loadu_si128((__m128i*) & image[i]);
		pixelMatrixR[index] = _mm_extract_epi8(mat, 0);
		pixelMatrixG[index] = _mm_extract_epi8(mat, 1);
		pixelMatrixB[index] = _mm_extract_epi8(mat, 2);
		index++;
		pixelMatrixR[index] = _mm_extract_epi8(mat, 4);
		pixelMatrixG[index] = _mm_extract_epi8(mat, 5);
		pixelMatrixB[index] = _mm_extract_epi8(mat, 6);
		index++;
		pixelMatrixR[index] = _mm_extract_epi8(mat, 8);
		pixelMatrixG[index] = _mm_extract_epi8(mat, 9);
		pixelMatrixB[index] = _mm_extract_epi8(mat, 10);
		index++;
		pixelMatrixR[index] = _mm_extract_epi8(mat, 12);
		pixelMatrixG[index] = _mm_extract_epi8(mat, 13);
		pixelMatrixB[index] = _mm_extract_epi8(mat, 14);
		index++;
	}

	int yAxis;
	int xAxis;

	vector<unsigned char> matrixR(255);
	vector<unsigned char> matrixG(255);
	vector<unsigned char> matrixB(255);

	__m128i matR;
	__m128i matG;
	__m128i matB;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			yAxis = y < 7 ? 7 : y > height - 8 ? height - 8 : y;
			xAxis = x < 7 ? 7 : x > width - 8 ? width - 8 : x;
			yAxis -= 7; xAxis -= 7;
			int index = 0;
			int index1 = xAxis;

			yAxis = y < 7 ? 7 : y > height - 8 ? height - 8 : y;
			xAxis = x < 7 ? 7 : x > width - 8 ? width - 8 : x;
			yAxis -= 7; xAxis -= 7;
			for (int i = 0; i < 15; i++) {
				matR = _mm_load_si128((__m128i*) & pixelMatrixR[((yAxis + i) * height) + xAxis]);
				matG = _mm_load_si128((__m128i*) & pixelMatrixG[((yAxis + i) * height) + xAxis]);
				matB = _mm_load_si128((__m128i*) & pixelMatrixB[((yAxis + i) * height) + xAxis]);
				_mm_store_si128((__m128i*) & matrixR[15 * i], matR);
				_mm_store_si128((__m128i*) & matrixG[15 * i], matG);
				_mm_store_si128((__m128i*) & matrixB[15 * i], matB);
			}

			nth_element(matrixR.begin(), matrixR.begin() + (225 / 2), matrixR.end());
			nth_element(matrixG.begin(), matrixG.begin() + (225 / 2), matrixG.end());
			nth_element(matrixB.begin(), matrixB.begin() + (225 / 2), matrixB.end());

			image[y * height * 4 + x * 4] = matrixR[225 / 2]; 
			image[y * height * 4 + x * 4 + 1] = matrixG[225 / 2]; 
			image[y * height * 4 + x * 4 + 2] = matrixB[225 / 2];
		}
	}
	return image;
}

int main()
{
	unsigned int sum = 0;
	for (int i = 0; i < 100; i++) {


		unsigned int start_time = clock();
		vector<unsigned char> png;
		vector<unsigned char> image; //the raw pixels
		unsigned width, height;

		//load and decode
		unsigned error = lodepng::load_file(png, "2400x2400.png");
		if (!error) error = lodepng::decode(image, width, height, png);

		image = MedianFilterOMP(image, width, height);

		encodeOneStep("exdee.png", image, width, height);
		unsigned int end_time = clock(); // конечное время
		//cout << end_time - start_time; // искомое время
		sum += end_time - start_time;
	}
	cout << sum / 5;
}