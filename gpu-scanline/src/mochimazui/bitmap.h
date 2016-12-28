
#ifndef _MOCHIMAZUI_BITMAP_H_
#define _MOCHIMAZUI_BITMAP_H_

#include <cstdint>
#include <string>
#include <vector>

#include <glm/vec4.hpp>

#include "color.h"

namespace Mochimazui {

	struct Bitmap {

	public:
		//Bitmap() {}
		//~Bitmap() {}

	public:
		//void load(const std::string &fileName);
		bool save(const std::string &fileName);

		void fill(const u8rgba &c) {
			for (uint32_t i = 0; i < _height; ++i) {
				for (uint32_t j = 0; j < _width; ++j) {
					pixel(j, i) = c;
				}
			}
		}

		void resize(int w, int h) {
			_width = w;
			_height = h;
			_pixel.resize(h*w);
		}

		uint32_t width() { return _width; }
		uint32_t height() { return _height; }

		const unsigned char * data() { return (const unsigned char*)_pixel.data(); }

	public:

		void setPixel(int x, int y, const u8rgba &color) {
			if (0 <= x && x < (int)_width && 0 <= y && y < (int)_height) {
				_pixel[y * _width + x] = color;
			}
		}

		u8rgba &pixel(int x, int y) {
			return _pixel[y * _width + x];
		}

		const u8rgba &pixel(int x, int y) const {
			return _pixel[y * _width + x];
		}

	private:
		uint32_t _width = 0;
		uint32_t _height = 0;
		std::vector<u8rgba> _pixel;
	};

}

#endif

