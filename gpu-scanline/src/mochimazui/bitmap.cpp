
#include "bitmap.h"

#include <cstdint>

#if defined _WIN32 || defined _WIN64
#include <Windows.h>
#endif

namespace Mochimazui {

	using std::string;

#if defined _WIN32 || defined _WIN64
#else
	typedef int32_t WORD;
	typedef int64_t DWORD;
	typedef int64_t LONG;

	typedef struct tagBITMAPFILEHEADER {
		WORD  bfType;
		DWORD bfSize;
		WORD  bfReserved1;
		WORD  bfReserved2;
		DWORD bfOffBits;
	} BITMAPFILEHEADER, *PBITMAPFILEHEADER;

	typedef struct tagBITMAPINFOHEADER {
		DWORD biSize;
		LONG  biWidth;
		LONG  biHeight;
		WORD  biPlanes;
		WORD  biBitCount;
		DWORD biCompression;
		DWORD biSizeImage;
		LONG  biXPelsPerMeter;
		LONG  biYPelsPerMeter;
		DWORD biClrUsed;
		DWORD biClrImportant;
	} BITMAPINFOHEADER, *PBITMAPINFOHEADER;
#endif

	//void Bitmap::load(const std::string &fileName) {
	//}

	bool Bitmap::save(const std::string &fileName) {

		BITMAPFILEHEADER bfh;
		BITMAPINFOHEADER bih;

		// fill info header
		bih.biSize = sizeof(BITMAPINFOHEADER);

		bih.biWidth = _width;
		bih.biHeight = _height;
		bih.biPlanes = 1;

		bih.biBitCount = 32;
		bih.biCompression = 0;
		bih.biSizeImage = _width*_height * 4;

		bih.biXPelsPerMeter = 1;
		bih.biYPelsPerMeter = 1;

		bih.biClrUsed = 0;
		bih.biClrImportant = 0;

		// fill file header
		bfh.bfType = 0x4D42;
		bfh.bfReserved1 = 0;
		bfh.bfReserved2 = 0;
		bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
		bfh.bfSize = bfh.bfOffBits + bih.biSizeImage;

		FILE* hFile;
#ifdef _MSC_VER
		fopen_s(&hFile, fileName.c_str(), "wb");
#else
		hFile = fopen(fileName.c_str(), "wb");
#endif
		if (!hFile) { return false; }

		fwrite(&bfh, sizeof(BITMAPFILEHEADER), 1, hFile);
		fwrite(&bih, sizeof(BITMAPINFOHEADER), 1, hFile);

		auto outputPixel = _pixel;
		//for (uint32_t y = 0; y < _height; ++y) {
		//	auto y0 = y;
		//	auto y1 = _height - y0 - 1;
		//	for (uint32_t x = 0; x < _width; ++x) {
		//		outputPixel[y0 * _width + x] = _pixel[y1 * _width + x];
		//	}
		//}

		fwrite(outputPixel.data(), bih.biSizeImage, 1, hFile);
		fclose(hFile);

		return true;
	}

}
