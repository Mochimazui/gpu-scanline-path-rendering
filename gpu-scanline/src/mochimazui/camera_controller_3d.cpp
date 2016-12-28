
#include "camera_controller_3d.h"

namespace Mochimazui {

	void CameraController3D::init(int width, int height)  {
		_sceneWidth = width;
		_sceneHeight = height;
		_rotateCenter = glm::vec3(width / 2.f, height / 2.f, 0.f);
		_rotateNormal = glm::vec3(0.f, 0.f, 1.f);
		_walkx = glm::vec3(1.f, 0.f, 0.f);
		_walky = glm::vec3(0.f, -1.f, 0.f);
	}

	void CameraController3D::fitToView(int vWidth, int vHeight) {

		//vWidth += 1;
		//vHeight += 1;

		Camera3D::reset();

		float sw = (float)vWidth / (float)_sceneWidth;
		float sh = (float)vHeight / (float)_sceneHeight;
		float s = sw < sh ? sw : sh;
		Camera3D::scale(glm::vec3(s, s, s));

		float nw = _sceneWidth * s;
		float nh = _sceneHeight * s;

		float dw = abs(vWidth - nw) *.5f;
		float dh = abs(vHeight - nh) * .5f;

		Camera3D::translate(glm::vec3(dw, dh, 0));

		init(vWidth, vHeight);
	}

	//
	glm::mat4x4 CameraController3D::projectionMatrix() {

		float W = (float)_sceneWidth;
		float H = (float)_sceneHeight;
		float W2 = W / 2.f;
		float H2 = H / 2.f;

		float S = W < H ? W : H;

		auto mat =
			glm::mat4x4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			W2, H2, -S, 1
			)
			*
			glm::mat4x4(
			S, 0, 0, 0,
			0, S, 0, 0,
			0, 0, 1, 1,
			0, 0, 0, 0
			)
			*
			glm::mat4x4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-W2, -H2, S, 1
			);

		return mat;
	}

	//
	glm::mat4x4 CameraController3D::modelViewMatrix() {
		return Camera3D::matrix();
	}

	void CameraController3D::leftButtonDown(int x, int y) {
		_leftButton = true;
		_lastPos = _leftButtonClickPos = ivec2(x, y);
	}

	void CameraController3D::leftButtonUp(int x, int y) {
		_leftButton = false;
	}

	void CameraController3D::rightButtonDown(int x, int y) {
		_rightButton = true;
		_lastPos = _rightButtonClickPos = ivec2(x, y);
	}

	void CameraController3D::rightButtonUp(int x, int y) {
		_rightButton = false;
	}

	void CameraController3D::move(int x, int y) {
		switch (_controllerMode) {
		case CCM_NULL:
			break;
		case CCM_MOVE:
			handleMove(x, y);
			break;
		case CCM_TURN:
			handleTurn(x, y);
			break;
		case CCM_ROTATE:
			handleRotate(x, y);
			break;
		default:
			throw std::runtime_error("CameraController3D::unsupported controller mode");
		}
	}

	// -------- -------- -------- -------- -------- -------- -------- --------
	void CameraController3D::handleMove(int x, int y) {
		ivec2 cp(x, y);
		if (_leftButton || _rightButton) {
			ivec2 delta = cp - _lastPos;
			delta.y *= -1;
			Camera3D::translate(delta.x * _walkx + delta.y * _walky);
		}
		_lastPos = cp;
	}

	void CameraController3D::handleTurn(int x, int y) {

		if (!_leftButton) {	return;}

		static const double RV = 0.01;
		ivec2 cp(x, y);
		glm::vec2 delta = cp - _lastPos;
		//delta.y *= -1;
		delta *= RV;

		auto tv3 = [](const glm::mat4x4& m, glm::vec3 &r3) {
			auto r4 = glm::vec4(r3.x, r3.y, r3.z, 0.f);
			r4 = m * r4;
			r4 = glm::normalize(r4);
			r3 = glm::vec3(r4.x, r4.y, r4.z);
		};

		Camera3D::turn_y(delta.y, _rotateCenter.y);
		auto m = glm::rotate(delta.y, glm::vec3(1, 0, 0));
		tv3(m, _rotateNormal);
		tv3(m, _walkx);
		tv3(m, _walky);

		_lastPos = cp;
	}

	void CameraController3D::handleRotate(int x, int y) {

		if (!_leftButton) { return; }

		static const double RV = 0.001;
		ivec2 cp(x, y);
		glm::vec2 delta = cp - _lastPos;
		delta.y *= -1;
		delta *= RV;

		auto tv3 = [](const glm::mat4x4& m, glm::vec3 &r3) {
			auto r4 = glm::vec4(r3.x, r3.y, r3.z, 0.f);
			r4 = m * r4;
			r4 = glm::normalize(r4);
			r3 = glm::vec3(r4.x, r4.y, r4.z);
		};

		Camera3D::rotate_cn(delta.x, _rotateCenter, _rotateNormal);
		auto m = glm::rotate(delta.x, _rotateNormal);

		_lastPos = cp;
	}

}