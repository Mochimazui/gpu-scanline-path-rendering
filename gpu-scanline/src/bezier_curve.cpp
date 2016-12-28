
#include "bezier_curve.h"

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <algorithm>

namespace Mochimazui {

using std::vector;

float2 SimpleBezierCurve::point(float t) const {
	if (!is_offset_curve()) {
		switch (_type) {
		default:
			throw std::runtime_error("SimpleBezierCurve: illegal curve type");
			break;
		case CT_Linear:
			return bezier_curve_point(_v[0], _v[1], t);
			break;
		case CT_Quadratic:
			return bezier_curve_point(_v[0], _v[1], _v[2], t);
			break;
		case CT_Cubic:
			return bezier_curve_point(_v[0], _v[1], _v[2], _v[3], t);
			break;
		case CT_Rational:
			return bezier_curve_point(_v[0], _v[1], _v[2], _v[3].x, t);
			break;
		}
	}
	else {
		switch (_type) {
		default:
			throw std::runtime_error("SimpleBezierCurve: illegal curve type");
			break;
		case CT_Linear:
			return offset_bezier_curve_point(_v[0], _v[1], _offset, t);
			break;
		case CT_Quadratic:
			return offset_bezier_curve_point(_v[0], _v[1], _v[2], _offset, t);
			break;
		case CT_Cubic:
			return offset_bezier_curve_point(_v[0], _v[1], _v[2], _v[3], _offset, t);
			break;
		case CT_Rational:
			return offset_bezier_curve_point(_v[0], _v[1], _v[2], _v[3].x, _offset, t);
			break;
		}
	}
}

float2 SimpleBezierCurve::tangent(float t) const {
	switch (_type) {
	default:
		throw std::runtime_error("SimpleBezierCurve: illegal curve type");
		break;
	case CT_Linear:
		return bezier_curve_tangent(_v[0], _v[1]);
		break;
	case CT_Quadratic:
		return bezier_curve_tangent(_v[0], _v[1], _v[2], t);
		break;
	case CT_Cubic:
		return bezier_curve_tangent(_v[0], _v[1], _v[2], _v[3], t);
		break;
	case CT_Rational:
		return bezier_curve_tangent(_v[0], _v[1], _v[2], _v[3].x, t);
		break;
	}
}

float2 SimpleBezierCurve::normal(float t) const {
	throw std::runtime_error("not implemented");
	return make_float2(0.f, 0.f);
}

SimpleBezierCurve SimpleBezierCurve::left(float t)  const {
	return subcurve(0.f, t);
}

SimpleBezierCurve SimpleBezierCurve::right(float t)  const {
	return subcurve(t, 1.f);
}

SimpleBezierCurve SimpleBezierCurve::subcurve(float t0, float t1)  const {

	SimpleBezierCurve c;

	c._type = _type;
	c._offset = _offset;

	switch (_type) {
	default:
		throw std::runtime_error("SimpleBezierCurve: illegal curve type");
		break;
	case CT_Linear:
		d_subcurve<CT_Linear>(
			_v[0], _v[1], _v[2], _v[3], _v[3].x,
			t0, t1,
			c._v[0], c._v[1], c._v[2], c._v[3], c._v[3].x);
		break;
	case CT_Quadratic:
		d_subcurve<CT_Quadratic>(
			_v[0], _v[1], _v[2], _v[3], _v[3].x,
			t0, t1,
			c._v[0], c._v[1], c._v[2], c._v[3], c._v[3].x);
		break;
	case CT_Cubic:
		d_subcurve<CT_Cubic>(
			_v[0], _v[1], _v[2], _v[3], _v[3].x,
			t0, t1,
			c._v[0], c._v[1], c._v[2], c._v[3], c._v[3].x);
		break;
	case CT_Rational:
		d_subcurve<CT_Rational>(
			_v[0], _v[1], _v[2], _v[3], _v[3].x,
			t0, t1,
			c._v[0], c._v[1], c._v[2], c._v[3], c._v[3].x);
		break;
	}

	return c;
}

float SimpleBezierCurve::line_manhattan_length()  const {
	float2 d = is_offset_curve() ? point(1.f) - point(0.f) : back() - front();
	return abs(d.x) + abs(d.y);
}

float SimpleBezierCurve::line_length()  const {
	float2 d = is_offset_curve() ? point(1.f) - point(0.f) : back() - front();
	return sqrt(d.x * d.x + d.y * d.y);
}

float SimpleBezierCurve::arc_length()  const {
	throw std::runtime_error("not implemented");
	return 0.f;
}

//float SimpleBezierCurve::control_point_line_length() const {
//}

void SimpleBezierCurve::subdiv(float len, std::function<void(float, float)> op_func)  const {
	std::function<void(float t0, float t1)> subdiv_func;
	subdiv_func = [this, len, &op_func, &subdiv_func](float t0, float t1) {

		auto tm = (t0 + t1) * .5f;

		auto p0 = this->point(t0);
		auto pm = this->point(tm);
		auto p1 = this->point(t1);

		float d = 0.f;
		d = std::max(d, abs(pm.x - p0.x));
		d = std::max(d, abs(pm.y - p0.y));
		d = std::max(d, abs(p1.x - p0.x));
		d = std::max(d, abs(p1.y - p0.y));

		if (d < len) { op_func(t0, t1); return; } 
		
		subdiv_func(t0, tm);
		subdiv_func(tm, t1);
	};
	subdiv_func(0.f, 1.f);
}

vector<SimpleBezierCurve> SimpleBezierCurve::subdiv_curve(float len)  const {
	vector<SimpleBezierCurve> sub_curves;
	subdiv(len, [this, &sub_curves](float t0, float t1) {
		sub_curves.push_back(this->subcurve(t0, t1));
	});
	return sub_curves;
}

vector<float> SimpleBezierCurve::subdiv_t(float len) const {
	vector<float> ts;
	subdiv(len, [&ts](float t0, float t1) {
		ts.push_back(t0);
	});
	ts.push_back(1.f);
	return ts;
}

SimpleBezierCurve SimpleBezierCurve::approxOffsetCurve() {

	if (_offset == 0.f) { return *this; }

	SimpleBezierCurve new_curve;

	new_curve._type = _type;
	new_curve._offset = 0;

	auto solve_intersection = [](float2 p0, float2 d0, float2 p2, float2 d2) {

		glm::mat2x2 A;
		A[0][0] = d0.x;
		A[0][1] = d0.y;

		A[1][0] = d2.x;
		A[1][1] = d2.y;

		glm::vec2 Y;
		Y[0] = p2.x - p0.x;
		Y[1] = p2.y - p0.y;

		glm::vec2 X;
		X = glm::inverse(A) * Y;

		if (X.x != X.x || X.y != X.y) {
			return make_float2(X.x, X.y);
		}

		float2 p1;
		p1.x = p0.x + X.x * d0.x;
		p1.y = p0.y + X.x * d0.y;

		return p1;
	};

	// -------- -------- -------- --------
	if (_type == CT_Linear) {
		new_curve._v[0] = point(0.f);
		new_curve._v[1] = point(1.f);
	}
	else if (_type == CT_Quadratic || _type==CT_Rational) {

		auto d0 = _v[1] - _v[0];
		auto d2 = _v[1] - _v[2];

		auto p0 = point(0.f);
		auto p2 = point(1.f);

		auto p1 = solve_intersection(p0, d0, p2, d2);
		if (p1.x != p1.x || p1.y != p1.y) { 
			p1 = point(0.5f); 
		}

		new_curve._v[0] = p0;
		new_curve._v[1] = p1;
		new_curve._v[2] = p2;
		new_curve._v[3] = _v[3];
	}
	else if (_type == CT_Cubic) {

		if (_v[0] == _v[1]) {

			SimpleBezierCurve mid_line;
			mid_line._type = CT_Linear;
			mid_line._offset = _offset;
			mid_line._v[0] = _v[1];
			mid_line._v[1] = _v[2];

			auto d0 = _v[2] - _v[1];
			auto d2 = _v[2] - _v[3];

			auto p0 = point(0.f);
			auto p2 = point(1.f);

			auto p1 = solve_intersection(p0, d0, p2, d2);
			if (p1.x != p1.x || p1.y != p1.y) { 
				new_curve._type = CT_Linear;
				new_curve._v[0] = p0;
				new_curve._v[1] = p2;
			}
			else {
				new_curve._v[0] = p0;
				new_curve._v[1] = p0;
				new_curve._v[2] = p1;
				new_curve._v[3] = p2;
			}

		}
		else if (_v[1] == _v[2]) {

			auto d0 = _v[1] - _v[0];
			auto d2 = _v[2] - _v[3];

			auto p0 = point(0.f);
			auto p2 = point(1.f);

			auto p1 = solve_intersection(p0, d0, p2, d2);
			if (p1.x != p1.x || p1.y != p1.y) {
				p1 = point(.5f);
			}
			else {
				new_curve._v[0] = p0;
				new_curve._v[1] = p1;
				new_curve._v[2] = p1;
				new_curve._v[3] = p2;
			}

		}
		else if (_v[2] == _v[3]) {

			SimpleBezierCurve mid_line;
			mid_line._type = CT_Linear;
			mid_line._offset = _offset;
			mid_line._v[0] = _v[1];
			mid_line._v[1] = _v[2];

			auto d0 = _v[1] - _v[0];
			auto d2 = _v[1] - _v[3];

			auto p0 = point(0.f);
			auto p2 = point(1.f);

			auto p1 = solve_intersection(p0, d0, p2, d2);
			if (p1.x != p1.x || p1.y != p1.y) {
				new_curve._type = CT_Linear;
				new_curve._v[0] = p0;
				new_curve._v[1] = p2;
			}
			else {
				new_curve._v[0] = p0;
				new_curve._v[1] = p1;
				new_curve._v[2] = p2;
				new_curve._v[3] = p2;
			}

		}
		else
		{
			SimpleBezierCurve mid_line;
			mid_line._type = CT_Linear;
			mid_line._offset = _offset;
			mid_line._v[0] = _v[1];
			mid_line._v[1] = _v[2];

			auto m_v1 = mid_line.point(0.f);
			auto m_v2 = mid_line.point(1.f);

			//
			auto p0 = point(0.f);
			auto d0 = _v[1] - _v[0];

			auto p3 = point(1.f);
			auto d3 = _v[2] - _v[3];

			auto p1 = solve_intersection(p0, d0, m_v2, m_v1 - m_v2);
			auto p2 = solve_intersection(p3, d3, m_v1, m_v2 - m_v1);

			if (p1.x != p1.x || p1.y != p1.y) { 
				p1 = mid_line.point(0.f);
			}
			if (p2.x != p2.x || p2.y != p2.y) { 
				p2 = mid_line.point(1.f);
			}

			new_curve._v[0] = p0;
			new_curve._v[1] = p1;
			new_curve._v[2] = p2;
			new_curve._v[3] = p3;
		}

	}
	else {
		throw std::runtime_error("SimpleBezierCurve:: invalid curve type");
	}

	return new_curve;
}

SimpleBezierCurve SimpleBezierCurve::approxOffsetCurve(float2 v_first, float2 v_last) {

	if (_offset == 0.f) { return *this; }

	SimpleBezierCurve new_curve;

	new_curve._type = _type;
	new_curve._offset = 0;

	auto solve_intersection = [](float2 p0, float2 d0, float2 p2, float2 d2) {

		glm::mat2x2 A;
		A[0][0] = d0.x;
		A[0][1] = d0.y;

		A[1][0] = d2.x;
		A[1][1] = d2.y;

		glm::vec2 Y;
		Y[0] = p2.x - p0.x;
		Y[1] = p2.y - p0.y;

		glm::vec2 X;
		X = glm::inverse(A) * Y;

		if (X.x != X.x || X.y != X.y) {
			return make_float2(X.x, X.y);
		}

		float2 p1;
		p1.x = p0.x + X.x * d0.x;
		p1.y = p0.y + X.x * d0.y;

		return p1;
	};

	// -------- -------- -------- --------
	if (_type == CT_Linear) {
		new_curve._v[0] = v_first;
		new_curve._v[1] = v_last;
	}
	else if (_type == CT_Quadratic || _type == CT_Rational) {

		auto d0 = _v[1] - _v[0];
		auto d2 = _v[1] - _v[2];

		auto p0 = v_first;
		auto p2 = v_last;

		auto p1 = solve_intersection(p0, d0, p2, d2);
		if (p1.x != p1.x || p1.y != p1.y) {
			new_curve._type = CT_Linear;
			new_curve._v[0] = p0;
			new_curve._v[1] = p2;
		}

		new_curve._v[0] = p0;
		new_curve._v[1] = p1;
		new_curve._v[2] = p2;
		new_curve._v[3] = _v[3];
	}
	else if (_type == CT_Cubic) {

		if (_v[0] == _v[1]) {

			SimpleBezierCurve mid_line;
			mid_line._type = CT_Linear;
			mid_line._offset = _offset;
			mid_line._v[0] = _v[1];
			mid_line._v[1] = _v[2];

			auto d0 = _v[2] - _v[1];
			auto d2 = _v[2] - _v[3];

			auto p0 = v_first;
			auto p2 = v_last;

			auto p1 = solve_intersection(p0, d0, p2, d2);
			if (p1.x != p1.x || p1.y != p1.y) {
				new_curve._type = CT_Linear;
				new_curve._v[0] = p0;
				new_curve._v[1] = p2;
			}
			else {
				new_curve._v[0] = p0;
				new_curve._v[1] = p0;
				new_curve._v[2] = p1;
				new_curve._v[3] = p2;
			}

		}
		else if (_v[1] == _v[2]) {

			auto d0 = _v[1] - _v[0];
			auto d2 = _v[2] - _v[3];

			auto p0 = v_first;
			auto p2 = v_last;

			auto p1 = solve_intersection(p0, d0, p2, d2);
			if (p1.x != p1.x || p1.y != p1.y) {
				new_curve._type = CT_Linear;
				new_curve._v[0] = p0;
				new_curve._v[1] = p2;
			}
			else {
				new_curve._v[0] = p0;
				new_curve._v[1] = p1;
				new_curve._v[2] = p1;
				new_curve._v[3] = p2;
			}

		}
		else if (_v[2] == _v[3]) {

			SimpleBezierCurve mid_line;
			mid_line._type = CT_Linear;
			mid_line._offset = _offset;
			mid_line._v[0] = _v[1];
			mid_line._v[1] = _v[2];

			auto d0 = _v[1] - _v[0];
			auto d2 = _v[1] - _v[3];

			auto p0 = v_first;
			auto p2 = v_last;

			auto p1 = solve_intersection(p0, d0, p2, d2);
			if (p1.x != p1.x || p1.y != p1.y) {
				new_curve._type = CT_Linear;
				new_curve._v[0] = p0;
				new_curve._v[1] = p2;
			}
			else {
				new_curve._v[0] = p0;
				new_curve._v[1] = p1;
				new_curve._v[2] = p2;
				new_curve._v[3] = p2;
			}

		}
		else
		{
			SimpleBezierCurve mid_line;
			mid_line._type = CT_Linear;
			mid_line._offset = _offset;
			mid_line._v[0] = _v[1];
			mid_line._v[1] = _v[2];

			auto m_v1 = mid_line.point(0.f);
			auto m_v2 = mid_line.point(1.f);

			//
			auto p0 = v_first;
			auto d0 = _v[1] - _v[0];

			auto p3 = v_last;
			auto d3 = _v[2] - _v[3];

			auto p1 = solve_intersection(p0, d0, m_v2, m_v1 - m_v2);
			auto p2 = solve_intersection(p3, d3, m_v1, m_v2 - m_v1);

			if (p1.x != p1.x || p1.y != p1.y) {
				p1 = mid_line.point(0.f);
			}
			if (p2.x != p2.x || p2.y != p2.y) {
				p2 = mid_line.point(1.f);
			}

			new_curve._v[0] = p0;
			new_curve._v[1] = p1;
			new_curve._v[2] = p2;
			new_curve._v[3] = p3;
		}

	}
	else {
		throw std::runtime_error("SimpleBezierCurve:: invalid curve type");
	}

	return new_curve;
}

vector<SimpleBezierCurve> SimpleBezierCurve::segApproxOffsetCurve() {

	static vector<SimpleBezierCurve> acurves;
	if (acurves.capacity() < 128) {
		acurves.reserve(128);
	}
	acurves.clear();

	if (_offset == 0.f) {
		acurves.push_back(*this);
		return acurves;
	}

	// recursive subdiv.
	if (_type == CT_Linear) {
		acurves.push_back(approxOffsetCurve());
		return acurves;
	}

	int max_depth = 0;

	// other curve type
	std::function<void(float, float, int)> subdiv;
	subdiv = [&](float left, float right, int depth) {

		max_depth = std::max(max_depth, depth);

		auto sub_curve = this->subcurve(left, right);		

		auto approx = sub_curve.approxOffsetCurve( point(left), point(right) );
		
		auto p0 = sub_curve.point(.5f);
		auto p1 = approx.point(.5f);
		//if ( (length(p0 - p1) <  1 / 128.f) || (depth >= 8) ) {
		//if ((length(p0 - p1) <  1 / 128.f) || (depth >= 8)) {
		if ((length(p0 - p1) <  1 / 32.f) || (depth >= 8)) {
			acurves.push_back(approx);
			return;
		}

		auto m = (left + right) * .5f;
		subdiv(left, m, depth + 1);
		subdiv(m, right, depth + 1);
	};

	subdiv(0.f, 1.f, 1);

	//printf("Subidv: %d, depth: %d\n", (uint32_t)acurves.size(), max_depth);
	return acurves;
}

}

