#ifndef _MOCHIMAZUI_CAMERA_CONTROLLER_2D_
#define _MOCHIMAZUI_CAMERA_CONTROLLER_2D_

#include <map>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "camera_2d.h"

namespace Mochimazui {

	using glm::vec2;
	using glm::ivec2;
	using std::map;

	class CameraController2D : public Camera2D {

	public:

		void leftButtonDown(int x, int y) {
			_leftButton = true;
			_lastPos = _leftButtonClickPos = ivec2(x, y);
		}

		void leftButtonUp(int x, int y) {
			_leftButton = false;
		}

		void rightButtonDown(int x, int y) {
			_rightButton = true;
			_lastPos = _rightButtonClickPos = ivec2(x, y);
		}

		void rightButtonUp(int x, int y) {
			_rightButton = false;
		}

		void wheel(float dy) {
			wheel(0.f, dy);
		}

		void wheel(float dx, float dy) {
			if (dy > 0) {
				scale(1.1f, _lastPos);
			}
			else {
				scale(0.9f, _lastPos);
			}			
		}

		void move(int x, int y) {
			ivec2 cp(x, y);
			if (_leftButton || _rightButton) {
				auto delta = cp - _lastPos;
				Camera2D::translate(delta);
			}
			_lastPos = cp;
		}

		void keyDown(int key) {
			_keyMap[key] = true;
		}

		void keyUp(int key) {
			_keyMap[key] = false;
		}

	private:

	private:

		bool _leftButton = false;
		bool _rightButton = false;

		ivec2 _leftButtonClickPos;
		ivec2 _rightButtonClickPos;

		ivec2 _lastPos;

		map<int, bool> _keyMap;
	};
}

#endif