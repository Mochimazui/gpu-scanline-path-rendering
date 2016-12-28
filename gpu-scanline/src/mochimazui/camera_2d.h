#ifndef _MOCHIMAZUI_CAMERA_2D_H_
#define _MOCHIMAZUI_CAMERA_2D_H_

#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace Mochimazui {

	using glm::vec2;
	using glm::mat3x3;

	class Camera2D {

		friend class CameraController2D;

	public:
		Camera2D() {
		}

	public:

		void reset() {
			_matrix = mat3x3();
		}

		//
		void translate(const vec2 &t) {
			translate(t.x, t.y);
		}

		template <class T>
		void translate(const T &x, const T &y) {
			_matrix = mat3x3(
				1, 0, 0, 
				0, 1, 0,
				x, y, 1
				) * _matrix;
		}

		//
		template <class T>
		void scale(const T &sx, const T &sy, const vec2 &cp = vec2(0.f, 0.f)) {
			translate(-cp);
			_matrix = mat3x3(
				sx, 0, 0,
				0, sy, 0,
				0, 0, 1
				) * _matrix;
			translate(cp);
		}

		template <class T>
		void scale(const T &s, const vec2 &cp = vec2(0.f, 0.f)) {
			scale(s, s, cp);
		}

		void scale(const vec2 &s, const vec2 &cp = vec2(0.f, 0.f)) {
			scale(s.x, s.y, cp);
		}

		//
		void rotate(const vec2 &c, float a) {
		}

		//
		glm::mat3x3 matrix() {
			return _matrix;
		}

	private:

		vec2 _scale;
		vec2 _translate;

		glm::mat3x3 _matrix;
	};

}

#endif