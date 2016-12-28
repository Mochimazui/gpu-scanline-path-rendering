#ifndef _MOCHIMAZUI_CAMERA_3D_H_
#define _MOCHIMAZUI_CAMERA_3D_H_

#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace Mochimazui {

	class Camera3D {

		friend class CameraController2D;

	public:
		Camera3D() {
		}

	public:

		void reset() {
			_matrix = glm::mat4x4();
		}

		//
		void translate(const glm::vec3 &t) {
			translate(t.x, t.y, t.z);
		}

		template <class T>
		void translate(const T &x, const T &y, const T &z) {
			_matrix = glm::mat4x4(
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				x, y, z, 1
				) * _matrix;
		}

		//
		template <class T>
		void scale(const T &sx, const T &sy, const glm::vec3 &cp = glm::vec3(0.f, 0.f, 0.f)) {
			translate(-cp);
			_matrix = glm::mat4x4(
				sx, 0, 0, 0,
				0, sy, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
				) * _matrix;
			translate(cp);
		}

		template <class T>
		void scale(const T &s, const glm::vec3 &cp = glm::vec3(0.f, 0.f, 0.f)) {
			scale(s, s, cp);
		}

		void scale(const glm::vec3 &s, const glm::vec3 &cp = glm::vec3(0.f, 0.f, 0.f)) {
			scale(s.x, s.y, cp);
		}

		//
		void rotate(const glm::vec3 &c, float a) {
		}

		//
		glm::mat4x4 matrix();

		//
		//void walk(const float delta, bool fixvr = true);

		void turn_y(const float d, const float y) {
			_matrix =
				glm::translate(glm::vec3(0, y, 0)) *
				glm::rotate(d, glm::vec3(1, 0, 0)) *
				glm::translate(glm::vec3(0, -y, 0)) * 
				_matrix;
		}


		void rotate_cn(const float d, const glm::vec3 &center, const glm::vec3 &normal) {
			_matrix =
				glm::translate(center) *
				glm::rotate(d, normal) *
				glm::translate(-center) *
				_matrix;
		}

		//void pan(const float dx, const float dy);

		void rotate(const float dx, const float dy) {
			//
			glm::vec4 dir = glm::vec4(_eye - _center, 1.f);
			dir = glm::rotate(dir, -dx, glm::vec3(0, 1, 0));
			//dir /= dir.w;
			//_eye = _center + vec3(dir);

			//
			auto cd = glm::cross(_up, glm::vec3(dir));
			dir = glm::rotate(dir, -dy, cd);
			dir /= dir.w;
			_eye = _center + glm::vec3(dir);

			//_up = glm::cross(vec3(dir), cd);
		}

	private:

		glm::vec3 _eye;
		glm::vec3 _center;
		glm::vec3 _up;

		glm::vec3 _scale;
		glm::vec3 _translate;

		glm::mat4x4 _matrix;
	};

}

#endif