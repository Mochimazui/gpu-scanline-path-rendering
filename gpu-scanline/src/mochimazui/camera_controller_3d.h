#ifndef _MOCHIMAZUI_CAMERA_CONTROLLER_3D_
#define _MOCHIMAZUI_CAMERA_CONTROLLER_3D_

#include <map>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "camera_3d.h"

namespace Mochimazui {

	using glm::vec2;
	using glm::ivec2;
	using std::map;

	enum CameraControllerMode {
		CCM_NULL,
		CCM_MOVE,
		CCM_TURN,
		CCM_ROTATE,
		CCM_WALK,
	};

	class CameraController3D : public Camera3D {

	public:
		void setControllerMode(int nm) {
			_controllerMode = nm;
		}

	public:

		void init(int width, int height);
		void fitToView(int vWidth, int vHeight);

	public:
		void leftButtonDown(int x, int y);
		void leftButtonUp(int x, int y);

		void rightButtonDown(int x, int y);
		void rightButtonUp(int x, int y);

		void move(int x, int y);

		void wheel(float dy) {
			wheel(0.f, dy);
		}

		void wheel(float dx, float dy) {
			if (dy > 0) {
				scale(1.05f, glm::vec3(_lastPos.x, _lastPos.y, 0.0f));
			}
			else {
				scale(0.95f, glm::vec3(_lastPos.x, _lastPos.y, 0.0f));
			}
		}

		void keyDown(int key) {
			_keyMap[key] = true;
		}

		void keyUp(int key) {
			_keyMap[key] = false;
		}

		glm::mat4x4 modelViewMatrix();
		glm::mat4x4 projectionMatrix();

	private:

		void handleMove(int x, int y);
		void handleTurn(int x, int y);
		void handleRotate(int x, int y);

	public:

		int _controllerMode = CCM_MOVE;

		bool _leftButton = false;
		bool _rightButton = false;

		ivec2 _leftButtonClickPos;
		ivec2 _rightButtonClickPos;

		ivec2 _lastPos;

		int _sceneWidth;
		int _sceneHeight;

		map<int, bool> _keyMap;

		glm::vec3 _rotateCenter;
		glm::vec3 _rotateNormal;
		glm::vec3 _walky;
		glm::vec3 _walkx;

	};
}

#endif