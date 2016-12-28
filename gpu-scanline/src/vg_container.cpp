

#define _CRT_SECURE_NO_WARNINGS

#include "vg_container.h"

#include <cstdio>
#include <cstdlib>

#include <functional>

#include <mochimazui/string.h>
#include <mochimazui/stdio_ext.h>

namespace Mochimazui {

// -------- -------- -------- -------- -------- -------- -------- --------
uint32_t VGContainer::addVertex(int number) {
	auto oldSize = vertex.point.size();
	vertex.point.resize(oldSize + number);
	return (uint32_t)oldSize;
}

uint32_t VGContainer::addVertex(const glm::vec2& v) {
	auto oldSize = vertex.point.size();
	vertex.point.push_back(v);
	return (uint32_t)oldSize;
}

uint32_t VGContainer::addVertex(const glm::vec2& v0, const glm::vec2& v1) {
	auto oldSize = vertex.point.size();
	vertex.point.push_back(v0);
	vertex.point.push_back(v1);
	return (uint32_t)oldSize;
}

uint32_t VGContainer::addVertex(const glm::vec2& v0, const glm::vec2& v1, const glm::vec2 &v2) {
	auto oldSize = vertex.point.size();
	vertex.point.push_back(v0);
	vertex.point.push_back(v1);
	vertex.point.push_back(v2);
	return (uint32_t)oldSize;
}

uint32_t VGContainer::addVertex(const glm::vec2& v0, const glm::vec2& v1, const glm::vec2 &v2, const glm::vec2 &v3) {
	auto oldSize = vertex.point.size();
	vertex.point.push_back(v0);
	vertex.point.push_back(v1);
	vertex.point.push_back(v2);
	vertex.point.push_back(v3);
	return (uint32_t)oldSize;
}

uint32_t VGContainer::addCurve(VGCurveType ct) {
	return addCurve(ct, 0.f);
}

uint32_t VGContainer::addLinear(const glm::vec2 &v0, const glm::vec2 &v1) {
	return addLinear(v0, v1, 0.f);
}

uint32_t VGContainer::addQuadratic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2) {
	return addQuadratic(v0, v1, v2, 0.f);
}

uint32_t VGContainer::addCubic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3) {
	return addCubic(v0, v1, v2, v3, 0.f);
}

uint32_t VGContainer::addRational(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float w) {
	return addRational(v0, v1, v2, w, 0.f);
}

// accept float2 
uint32_t VGContainer::addLinear(const float2 &v0, const float2 &v1, float offset) {
	return addLinear(glm::vec2(v0.x, v0.y), glm::vec2(v1.x, v1.y), offset);
}
uint32_t VGContainer::addQuadratic(const float2 &v0, const float2 &v1, const float2 &v2, float offset) {
	return addQuadratic(glm::vec2(v0.x, v0.y), glm::vec2(v1.x, v1.y), glm::vec2(v2.x, v2.y), offset);
}
uint32_t VGContainer::addCubic(const float2 &v0, const float2 &v1, const float2 &v2, const float2 &v3, float offset) {
	return addCubic(glm::vec2(v0.x, v0.y), glm::vec2(v1.x, v1.y), glm::vec2(v2.x, v2.y), glm::vec2(v3.x, v3.y), offset);
}
uint32_t VGContainer::addRational(const float2 &v0, const float2 &v1, const float2 &v2, float w, float offset) {
	return addRational(glm::vec2(v0.x, v0.y), glm::vec2(v1.x, v1.y), glm::vec2(v2.x, v2.y), w, offset);
}

uint32_t VGContainer::addCurve(VGCurveType ct, float offset) {

	auto oldSize = curveNumber();
	curve.vertexIndex.push_back(vertexNumber());
	curve.type.push_back(ct);
	curve.reversed.push_back(false);
	curve.arc_w1s.push_back(0x7f7fffff);
	curve.offset.push_back(offset);

	++contour.curveNumber[contourNumber() - 1];

	return oldSize;
}

uint32_t VGContainer::addLinear(const glm::vec2 &v0, const glm::vec2 &v1, float offset) {
	auto c = addCurve(CT_Linear, offset);
	addVertex(v0, v1);
	return c;
}

uint32_t VGContainer::addQuadratic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float offset) {
	auto c = addCurve(CT_Quadratic, offset);
	addVertex(v0, v1, v2);
	return c;
}

uint32_t VGContainer::addCubic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3, float offset) {
	auto c = addCurve(CT_Cubic, offset);
	addVertex(v0, v1, v2, v3);
	return c;
}

uint32_t VGContainer::addRational(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float w, float offset) {
	auto c = addCurve(CT_Rational, offset);
	curve.arc_w1s.back() = *((int*)&w);
	addVertex(v0, v1, v2);
	return c;
}

uint32_t VGContainer::addCurve(const SimpleBezierCurve &bc) {
	if (bc.type() == CT_Linear) { 
		return addLinear(bc[0], bc[1], bc._offset);
	}
	else if (bc.type() == CT_Quadratic) {
		return addQuadratic(bc[0], bc[1], bc[2], bc._offset);
	}
	else if (bc.type() == CT_Cubic) {
		return addCubic(bc[0], bc[1], bc[2], bc[3], bc._offset);
	}
	else { 
		return addRational(bc[0], bc[1], bc[2], bc.w());
	}
}

void VGContainer::addSVGArc(
	glm::vec2 v_start, float i_rx, float i_ry,
	float x_axis_rotation, int large_arc_flag, int sweep_flag, glm::vec2 v_end) {

	// arc -> cubic
	// code from https://raw.githubusercontent.com/vvvv/SVG/master/Source/Paths/SvgArcSegment.cs

	const double PI = acos(-1.0);
	const double RadiansPerDegree = acos(-1.0) / 180.0;
	const double DoublePI = acos(-1.0) * 2;

	auto CalculateVectorAngle = [&](double ux, double uy, double vx, double vy) -> double {

		double ta = atan2(uy, ux);
		double tb = atan2(vy, vx);

		if (tb >= ta)
		{
			return tb - ta;
		}

		return DoublePI - (ta - tb);
	};

	auto Start = v_start;
	auto End = v_end;

	auto RadiusX = i_rx;
	auto RadiusY = i_ry;

	auto Size = large_arc_flag;
	auto Sweep = sweep_flag;

	if (RadiusX == 0.0f && RadiusY == 0.0f)
	{
		// add line and quit.
	}

	auto Angle = x_axis_rotation;

	double sinPhi = sin(Angle * RadiansPerDegree);
	double cosPhi = cos(Angle * RadiansPerDegree);

	double x1dash = cosPhi * (Start.x - End.x) / 2.0 + sinPhi * (Start.y - End.y) / 2.0;
	double y1dash = -sinPhi * (Start.x - End.x) / 2.0 + cosPhi * (Start.y - End.y) / 2.0;

	double root;
	double numerator = RadiusX * RadiusX * RadiusY * RadiusY - RadiusX * RadiusX * y1dash * y1dash - RadiusY * RadiusY * x1dash * x1dash;

	float rx = RadiusX;
	float ry = RadiusY;

	if (numerator < 0.0)
	{
		float s = (float)sqrt(1.0 - numerator / (RadiusX * RadiusX * RadiusY * RadiusY));

		rx *= s;
		ry *= s;
		root = 0.0;
	}
	else
	{
		root = (
			(Size == ARC_SIZE_LARGE && Sweep == ARC_SWEEP_POSITIVE)
			|| 
			(Size == ARC_SIZE_SMALL && Sweep == ARC_SWEEP_NEGATIVE) 
			? -1.0 : 1.0)
			* sqrt(numerator / (RadiusX * RadiusX * y1dash * y1dash + RadiusY * RadiusY * x1dash * x1dash));
	}

	double cxdash = root * rx * y1dash / ry;
	double cydash = -root * ry * x1dash / rx;

	double cx = cosPhi * cxdash - sinPhi * cydash + (Start.x + End.x) / 2.0;
	double cy = sinPhi * cxdash + cosPhi * cydash + (Start.y + End.y) / 2.0;

	double theta1 = CalculateVectorAngle(1.0, 0.0, (x1dash - cxdash) / rx, (y1dash - cydash) / ry);
	double dtheta = CalculateVectorAngle((x1dash - cxdash) / rx, (y1dash - cydash) / ry, (-x1dash - cxdash) / rx, (-y1dash - cydash) / ry);

	if (Sweep == ARC_SWEEP_NEGATIVE && dtheta > 0)
	{
		dtheta -= 2.0 * PI;
	}
	else if (Sweep == ARC_SWEEP_POSITIVE && dtheta < 0)
	{
		dtheta += 2.0 * PI;
	}

	int segments = (int)ceil((double)abs(dtheta / (PI / 2.0)));
	double delta = dtheta / segments;
	double t = 8.0 / 3.0 * sin(delta / 4.0) * sin(delta / 4.0) / sin(delta / 2.0);

	double startX = Start.x;
	double startY = Start.y;

	for (int i = 0; i < segments; ++i)
	{
		double cosTheta1 = cos(theta1);
		double sinTheta1 = sin(theta1);
		double theta2 = theta1 + delta;
		double cosTheta2 = cos(theta2);
		double sinTheta2 = sin(theta2);

		double endpointX = cosPhi * rx * cosTheta2 - sinPhi * ry * sinTheta2 + cx;
		double endpointY = sinPhi * rx * cosTheta2 + cosPhi * ry * sinTheta2 + cy;

		double dx1 = t * (-cosPhi * rx * sinTheta1 - sinPhi * ry * cosTheta1);
		double dy1 = t * (-sinPhi * rx * sinTheta1 + cosPhi * ry * cosTheta1);

		double dxe = t * (cosPhi * rx * sinTheta2 + sinPhi * ry * cosTheta2);
		double dye = t * (sinPhi * rx * sinTheta2 - cosPhi * ry * cosTheta2);

		addCurve(CT_Cubic);
		auto vi = addVertex(4);

		vertex.point[vi] = glm::vec2(startX, startY);
		vertex.point[vi + 1] = glm::vec2((float)(startX + dx1), (float)(startY + dy1));
		vertex.point[vi + 2] = glm::vec2((float)(endpointX + dxe), (float)(endpointY + dye));
		vertex.point[vi + 3] = glm::vec2(endpointX, endpointY);

		theta1 = theta2;
		startX = (float)endpointX;
		startY = (float)endpointY;
	}

}

uint32_t VGContainer::addContour() {
	auto oldSize = contourNumber();

	contour.curveIndex.push_back(curveNumber());
	contour.curveNumber.push_back(0);
	contour.closed.push_back(0);

	++path.contourNumber[pathNumber() - 1];

	return oldSize;
}

uint32_t VGContainer::addPath() {
	auto oldSize = pathNumber();

	path.contourIndex.push_back(contourNumber());
	path.contourNumber.push_back(0);

	path.fillType.push_back(FT_NONE);
	path.fillRule.push_back(FR_NON_ZERO);
	path.fillColor.push_back(u8rgba());
	path.fillIndex.push_back(0xFFFFFFFF);
	path.fillOpacity.push_back(1.f);

	path.strokeType.push_back(ST_NONE);
	path.strokeColor.push_back(u8rgba());
	path.strokeIndex.push_back(0xFFFFFFFF);
	path.strokeOpacity.push_back(1.f);


	path.strokeWidth.push_back(1.f);
	path.strokeLineCap.push_back(STROKE_CAP_BUTT);
	path.strokeLineJoin.push_back(STROKE_JOIN_MITER);
	path.strokeMiterLimit.push_back(4);

	path.gradientHref.push_back("");
	path.svgString.push_back("");

	return oldSize;
}

uint32_t VGContainer::addPath(int f_rule, int f_mode, u8rgba f_color, int f_index) {

	auto pid = addPath();

	path.fillRule[pid] = f_rule;
	path.fillType[pid] = f_mode;
	path.fillColor[pid] = f_color;
	path.fillIndex[pid] = f_index;

	return pid;
}

uint32_t VGContainer::addStrokePath(u8rgba s_color, float width) {

	auto pid = addPath();

	path.strokeType[pid] = ST_COLOR;
	path.strokeColor[pid] = s_color;
	path.strokeWidth[pid] = width;

	return pid;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGContainer::reduceDegenerate() {

}

void VGContainer::mergeAdjacentPath() {


	for (int i = 0; i + 1 < path.contourIndex.size();) {
		int j = i + 1;

		for (; j < path.contourIndex.size(); ++j) {

			if (
				(path.fillType[i] != path.fillType[j])
				|| (path.fillRule[i] != path.fillRule[j])
				|| (path.fillColor[i] != path.fillColor[j])
				|| (path.fillIndex[i] != path.fillIndex[j])
				|| (path.fillOpacity[i] != path.fillOpacity[j])
				) {
				break;
			}
		}

		for (int k = i + 1; k < j; ++k) {
			path.contourNumber[i] += path.contourNumber[k];
			path.svgString[i] += " Z " + path.svgString[k];
		}

		path.contourIndex.erase(path.contourIndex.begin() + i + 1, path.contourIndex.begin() + j);
		path.contourNumber.erase(path.contourNumber.begin() + i + 1, path.contourNumber.begin() + j);

		path.fillType.erase(path.fillType.begin() + i + 1, path.fillType.begin() + j);
		path.fillRule.erase(path.fillRule.begin() + i + 1, path.fillRule.begin() + j);
		path.fillColor.erase(path.fillColor.begin() + i + 1, path.fillColor.begin() + j);
		path.fillIndex.erase(path.fillIndex.begin() + i + 1, path.fillIndex.begin() + j);
		path.fillOpacity.erase(path.fillOpacity.begin() + i + 1, path.fillOpacity.begin() + j);

		path.strokeType.erase(path.strokeType.begin() + i + 1, path.strokeType.begin() + j);
		path.strokeColor.erase(path.strokeColor.begin() + i + 1, path.strokeColor.begin() + j);
		path.strokeIndex.erase(path.strokeIndex.begin() + i + 1, path.strokeIndex.begin() + j);

		//path.gradientIndex.erase(path.gradientIndex.begin() + i + 1);
		path.gradientHref.erase(path.gradientHref.begin() + i + 1, path.gradientHref.begin() + j);
		path.svgString.erase(path.svgString.begin() + i + 1, path.svgString.begin() + j);

		++i;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGContainer::reserve(int path, int contour, int curve, int vertex) {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGContainer::clear() {
	vertex.clear();
	curve.clear();
	contour.clear();
	path.clear();
	gradient.clear();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGContainer::cutToLength() {
	return;
	static const int cut_length = 32;

	if (!curveNumber()) { return; }

	auto cid = curveNumber() - 1;
	auto type = curve.type[cid];
	auto vpos = curve.vertexIndex[cid];
	auto vn = type & 7;

	auto vf = vertex.point[vpos];
	auto vl = vertex.point[vpos + vn - 1];

	auto d = vl - vf;

	auto len = sqrt(d.x * d.x + d.y * d.y);

	if (len < cut_length) { return; }

	int cut_n = (int)(len / cut_length);
	++cut_n;

	glm::vec2 iv0, iv1, iv2, iv3; float iw;
	glm::vec2 ov0, ov1, ov2, ov3; float ow;

	iv0 = vertex.point[vpos + 0];
	iv1 = vertex.point[vpos + 1];
	if (vn >= 3) { iv2 = vertex.point[vpos + 2]; }
	if (vn >= 4) { iv3 = vertex.point[vpos + 3]; }
	if (type == CT_Rational) {
		iw = *((float*)&curve.arc_w1s[cid]);
		iv1 *= iw;
	}

	auto lerp2 = [](const glm::vec2 &a, const  glm::vec2 &b, float t) {
		return a * (1 - t) + b*t;
	};

	auto sub_curve = [&](int type, float t0, float t1) {
		if (type == CT_Linear) {
			ov0 = lerp2(iv0, iv1, t0);
			ov1 = lerp2(iv0, iv1, t1);
		}
		else if (type == CT_Quadratic) {

			glm::vec2 la0 = lerp2(iv0, iv1, t0);
			glm::vec2 la1 = lerp2(iv1, iv2, t0);

			glm::vec2 lb0 = lerp2(iv0, iv1, t1);
			glm::vec2 lb1 = lerp2(iv1, iv2, t1);

			ov0 = lerp2(la0, la1, t0);
			ov1 = lerp2(la0, la1, t1);
			ov2 = lerp2(lb0, lb1, t1);
		}
		else if (type == CT_Cubic) {

			float a = t1;
			float b = t0 / t1;
			if (b != b) { b = 0.f; }

			// left 
			glm::vec2 c30 = lerp2(iv0, iv1, a);
			glm::vec2 c31 = lerp2(iv1, iv2, a);
			glm::vec2 c32 = lerp2(iv2, iv3, a);

			glm::vec2 c20 = lerp2(c30, c31, a);
			glm::vec2 c21 = lerp2(c31, c32, a);

			ov0 = iv0;
			ov1 = c30;
			ov2 = c20;
			ov3 = lerp2(c20, c21, a);

			// right
			c30 = lerp2(ov0, ov1, b);
			c31 = lerp2(ov1, ov2, b);
			c32 = lerp2(ov2, ov3, b);

			c20 = lerp2(c30, c31, b);
			c21 = lerp2(c31, c32, b);

			ov0 = lerp2(c20, c21, b);
			ov1 = c21;
			ov2 = c32;
		}
		else {
			auto blossom = [](glm::vec2 *B, float Bw, float u, float v, float &w) -> glm::vec2
			{
				float uv = u*v;
				float b0 = uv - u - v + 1,
					b1 = u + v - 2 * uv,
					b2 = uv;
				w = 1 * b0 + Bw*b1 + 1 * b2;
				return B[0] * b0 + B[1] * b1 + B[2] * b2;
			};

			float u = t0;
			float v = t1;

			glm::vec2 cB[3] = { iv0, iv1, iv2 };
			float cBw = iw;

			float wA, wB, wC;
			glm::vec2 A = blossom(cB, cBw, u, u, wA);
			glm::vec2 B = blossom(cB, cBw, u, v, wB);
			glm::vec2 C = blossom(cB, cBw, v, v, wC);

			float s = 1.0f / sqrt(wA * wC);
			ov1 = s*B;
			ow = s*wB;

			if (u == 0)
			{
				ov0 = cB[0];
				ov2 = C / wC;
			}
			else if (v == 1)
			{
				ov0 = A / wA;
				ov2 = cB[2];
			}
			else
			{
				ov0 = A / wA;
				ov2 = C / wC;
			}
		}
	};

	std::vector<float> cut_t;
	for (int i = 0; i <= cut_n; ++i) {
		cut_t.push_back(i / (float)cut_n);
	}

	// update current curve
	sub_curve(type, cut_t[0], cut_t[1]);
	vertex.point[vpos] = ov0;
	vertex.point[vpos + 1] = ov1;
	if (vn >= 3) { vertex.point[vpos + 2] = ov2; }
	if (vn >= 4) { vertex.point[vpos + 3] = ov3; }
	if (type == CT_Rational) {
		curve.arc_w1s[cid] = *((int*)&ow);
		vertex.point[vpos + 1] /= ow;
	}

	// add new curves.
	for (int i = 1; i < cut_n; ++i) {
		sub_curve(type, cut_t[i], cut_t[i + 1]);
		if (type == CT_Linear) { addLinear(ov0, ov1); }
		else if (type == CT_Quadratic) { addQuadratic(ov0, ov1, ov2); }
		else if (type == CT_Cubic) { addCubic(ov0, ov1, ov2, ov3); }
		else { addRational(ov0, ov1 / ow, ov2, ow); }
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
VGContainer VGContainer::strokeToOffsetCurve() {

	VGContainer new_vg;

	new_vg.x = x;
	new_vg.y = y;
	new_vg.width = width;
	new_vg.height = height;
	new_vg.vp_pos_x = vp_pos_x;
	new_vg.vp_pos_y = vp_pos_y;

	for (uint32_t pi = 0; pi < pathNumber(); ++pi) {

		// fill path
		auto ft = path.fillType[pi];
		if (ft != FT_NONE) {

			auto npi = new_vg.addPath();

			new_vg.path.fillType[npi] = path.fillType[pi];
			new_vg.path.fillColor[npi] = path.fillColor[pi];
			new_vg.path.fillIndex[npi] = path.fillIndex[pi];
			new_vg.path.fillOpacity[npi] = path.fillOpacity[pi];
			new_vg.path.fillRule[npi] = path.fillRule[pi];

			new_vg.path.strokeType[npi] = ST_NONE;

			auto contour_begin = path.contourIndex[pi];
			auto contour_n = path.contourNumber[pi];
			auto contour_end = contour_begin + contour_n;

			for (uint32_t cti = contour_begin; cti < contour_end; ++cti) {

				new_vg.addContour();

				auto curve_begin = contour.curveIndex[cti];
				auto curve_n = contour.curveNumber[cti];
				auto curve_end = curve_begin + curve_n;
				
				uint32_t contour_first_vertex_i = 0;
				uint32_t contour_last_vertex_i = 0;

				for (uint32_t cvi = curve_begin; cvi < curve_end; ++cvi) {

					auto curve_type = curve.type[cvi];
					auto vi = curve.vertexIndex[cvi];
					auto curve_reversed = curve.reversed[cvi];
					if (curve_reversed) {
						throw std::runtime_error("DO NOT SUPPORT REVERSED CURVE HERE");
					}

					if (cvi == curve_begin) { contour_first_vertex_i = vi; }
					contour_last_vertex_i = vi + (curve_type & 7) - 1;

					if (curve_type == CT_Linear) {
						new_vg.addLinear(
							vertex.point[vi],
							vertex.point[vi + 1]
							);
					}
					else if (curve_type == CT_Quadratic) {
						new_vg.addQuadratic(
							vertex.point[vi],
							vertex.point[vi + 1],
							vertex.point[vi + 2]
							);
					}
					else if (curve_type == CT_Cubic) {
						new_vg.addCubic(
							vertex.point[vi],
							vertex.point[vi + 1],
							vertex.point[vi + 2],
							vertex.point[vi + 3]
							);
					}
					else if (curve_type == CT_Rational) {
						new_vg.addRational(
							vertex.point[vi],
							vertex.point[vi + 1],
							vertex.point[vi + 2],
							*((float*)&curve.arc_w1s[cvi])
							);
					}
					else {
						throw std::runtime_error("VGContainer::strokeToOffsetCurve: illegal curve type");
					}
				}

				// always close filled path.
				if (contour_first_vertex_i < contour_last_vertex_i) {
					auto fv = vertex.point[contour_first_vertex_i];
					auto lv = vertex.point[contour_last_vertex_i];

					if (fv != lv) {
						new_vg.addLinear(lv, fv);
					}
				}

			}
		}

		// stroke to fill.
		auto st = path.strokeType[pi];
		if (st != ST_NONE) {

			auto npi = new_vg.addPath();
			auto stroke_type = path.strokeType[pi];

			new_vg.path.fillType[npi] = stroke_type == ST_COLOR ? FT_COLOR : FT_GRADIENT;
			new_vg.path.fillColor[npi] = path.strokeColor[pi];
			new_vg.path.fillIndex[npi] = path.strokeIndex[pi];
			new_vg.path.fillOpacity[npi] = path.strokeOpacity[pi];
			new_vg.path.fillRule[npi] = FR_NON_ZERO;

			new_vg.path.strokeType[npi] = ST_NONE;

			//
			auto stroke_width = path.strokeWidth[pi];
			auto half_stroke_width = stroke_width * .5f;

			auto stroke_line_cap = path.strokeLineCap[pi];
			auto stroke_line_join = path.strokeLineJoin[pi];
			auto stroke_miter_limit = path.strokeMiterLimit[pi];

			auto contour_begin = path.contourIndex[pi];
			auto contour_n = path.contourNumber[pi];
			auto contour_end = contour_begin + contour_n;

			for (uint32_t cti = contour_begin; cti < contour_end; ++cti) {

				new_vg.addContour();

				auto curve_begin = contour.curveIndex[cti];
				auto curve_n = contour.curveNumber[cti];
				auto curve_end = curve_begin + curve_n;
				auto contour_colsed = contour.closed[cti];

				// collect all end points and 

				std::vector<float2> end_point;
				std::vector<float2> pos_offset_point;
				std::vector<float2> neg_offset_point;

				// for 'miter' join we need tangent vector.
				std::vector<float2> pos_offset_tan_vector;
				std::vector<float2> neg_offset_tan_vector;

				// (1) 

				for (uint32_t cvi = curve_begin; cvi < curve_end; ++cvi) {

					auto curve_reversed = curve.reversed[cvi];
					if (curve_reversed) {
						throw std::runtime_error("DO NOT SUPPORT REVERSED CURVE HERE");
					}

					auto curve_type = curve.type[cvi];
					auto vi = curve.vertexIndex[cvi];

					float2 f2v[4];

					for (int i = 0; i < (curve_type & 7); ++i) {
						auto glmv = vertex.point[vi + i];
						f2v[i] = make_float2(glmv.x, glmv.y);
					}
					if (curve_type == CT_Rational) {
						f2v[3].x = *((float*)&curve.arc_w1s[cvi]);
					}

					SimpleBezierCurve bezier_curve;

					if (curve_type == CT_Linear) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], 0.f);
					}
					else if (curve_type == CT_Quadratic) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], 0.f);
					}
					else if (curve_type == CT_Cubic) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], f2v[3], 0.f);
					}
					else if (curve_type == CT_Rational) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], f2v[3].x, 0.f);
					}
					else {
						throw std::runtime_error("VGContainer::strokeToOffsetCurve: illegal curve type");
					}

					// record vertex & tangent vector for join.
					auto pos_bezier_curve = bezier_curve.offset(half_stroke_width);
					auto neg_bezier_curve = bezier_curve.offset(-half_stroke_width);

					end_point.push_back(bezier_curve.front());
					end_point.push_back(bezier_curve.back());

					pos_offset_point.push_back(pos_bezier_curve.point(0.f));
					pos_offset_point.push_back(pos_bezier_curve.point(1.f));

					neg_offset_point.push_back(neg_bezier_curve.point(0.f));
					neg_offset_point.push_back(neg_bezier_curve.point(1.f));

					pos_offset_tan_vector.push_back(-pos_bezier_curve.tangent(0.f));
					pos_offset_tan_vector.push_back(pos_bezier_curve.tangent(1.f));

					neg_offset_tan_vector.push_back(-neg_bezier_curve.tangent(0.f));
					neg_offset_tan_vector.push_back(neg_bezier_curve.tangent(1.f));

				} // end of curve FOR

				// (2) start cap

				if (contour_colsed) {
					// gen join

					auto vf = end_point.front();
					auto pvf = pos_offset_point.front();
					auto nvf = neg_offset_point.front();
					auto ptf = pos_offset_tan_vector.front();
					auto ntf = neg_offset_tan_vector.front();

					//
					new_vg.addLinear(nvf, pvf);

					//end_point.push_back(vf);
					//pos_offset_point.push_back(pvf);
					//neg_offset_point.push_back(nvf);
					//pos_offset_tan_vector.push_back(ptf);
					//neg_offset_tan_vector.push_back(ntf);
				}
				else {
					// gen cap

					auto l00 = pos_offset_point.front();
					auto l01 = neg_offset_point.front();

					auto l10 = pos_offset_point.back();
					auto l11 = neg_offset_point.back();

					new_vg.addLinear(glm::vec2(l01.x, l01.y), glm::vec2(l00.x, l00.y));
					//new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));

					/*
					if (stroke_line_cap == STROKE_CAP_BUTT) {

						auto l00 = pos_offset_point.front();
						auto l01 = neg_offset_point.front();

						auto l10 = pos_offset_point.back();
						auto l11 = neg_offset_point.back();

						new_vg.addLinear(glm::vec2(l01.x, l01.y), glm::vec2(l00.x, l00.y));
						//new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
					}
					else if (stroke_line_cap == STROKE_CAP_ROUND) {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: STROKE_CAP_ROUNT not implemented.");
					}
					else if (stroke_line_cap == STROKE_CAP_SQUARE) {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: STROKE_CAP_SQUARE not implemented.");
					}
					else {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: illegal stroke line cap");
					}
					*/
				}


				// (3) pos curves & join

				for (uint32_t cvi = curve_begin; cvi < curve_end; ++cvi) {

					auto curve_reversed = curve.reversed[cvi];
					if (curve_reversed) {
						throw std::runtime_error("DO NOT SUPPORT REVERSED CURVE HERE");
					}

					auto curve_type = curve.type[cvi];
					auto vi = curve.vertexIndex[cvi];

					float2 f2v[4];

					for (int i = 0; i < (curve_type & 7); ++i) {
						auto glmv = vertex.point[vi + i];
						f2v[i] = make_float2(glmv.x, glmv.y);
					}
					if (curve_type == CT_Rational) {
						f2v[3].x = *((float*)&curve.arc_w1s[cvi]);
					}

					SimpleBezierCurve bezier_curve;

					if (curve_type == CT_Linear) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], 0.f);
						new_vg.addLinear(vertex.point[vi], vertex.point[vi + 1], half_stroke_width);
					}
					else if (curve_type == CT_Quadratic) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], 0.f);
						new_vg.addQuadratic(vertex.point[vi], vertex.point[vi + 1],
							vertex.point[vi + 2], half_stroke_width);
					}
					else if (curve_type == CT_Cubic) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], f2v[3], 0.f);
						new_vg.addCubic(vertex.point[vi], vertex.point[vi + 1],
							vertex.point[vi + 2], vertex.point[vi + 3], half_stroke_width);
					}
					else if (curve_type == CT_Rational) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], f2v[3].x, 0.f);
						new_vg.addRational(vertex.point[vi], vertex.point[vi + 1],
							vertex.point[vi + 2], *((float*)&curve.arc_w1s[cvi]), half_stroke_width);
					}
					else {
						throw std::runtime_error("VGContainer::strokeToOffsetCurve: illegal curve type");
					}

					// join
					auto i = cvi * 2 + 1;
					if (i + 1< end_point.size()) {

						const auto &l00 = pos_offset_point[i];
						const auto &l01 = pos_offset_point[i + 1];

						const auto &t00 = pos_offset_tan_vector[i];
						const auto &t01 = pos_offset_tan_vector[i + 1];

						const auto &l10 = neg_offset_point[i + 1];
						const auto &l11 = neg_offset_point[i];

						const auto &t10 = neg_offset_tan_vector[i + 1];
						const auto &t11 = neg_offset_tan_vector[i];

						stroke_line_join = STROKE_JOIN_BEVEL;

						if (stroke_line_join == STROKE_JOIN_MITER) {

							// L00 + T00 * a = P0
							// L01 + T01 * b = P0

							//l00.x + t00.x * a = l01.x + t01.x * b;
							//l00.y + t00.y * a = l01.y + t01.y * b;

							//t00.x * a - t01.x * b = l01.x - l00.x;
							//t00.y * a - t01.y * b = l01.y - l00.y;

							glm::mat2x2 A;
							glm::vec2 Y, X0, X1;

							A[0][0] = t00.x;
							A[0][1] = t00.y;

							A[1][0] = t01.x;
							A[1][1] = t01.y;

							Y.x = l01.x - l00.x;
							Y.y = l01.y - l00.y;

							X0 = glm::inverse(A) * Y;

							float2 P0 = l00 + t00 * X0.x;

							// L10 + T10 * c = P1
							// L11 + T11 * d = P!

							// add line : L00-P0, P0-L01
							// add line : L10-P1, P1-L11

							A[0][0] = t10.x;
							A[0][1] = t10.y;

							A[1][0] = t11.x;
							A[1][1] = t11.y;

							Y.x = l11.x - l10.x;
							Y.y = l11.y - l10.y;

							X1 = glm::inverse(A) * Y;

							auto P1 = l10 + t10 * X1.x;

							auto dp = P1 - P0;
							auto d = sqrt(dp.x * dp.x + dp.y * dp.y);

							if (d / stroke_width * 2 > stroke_miter_limit) {
								new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
								//new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
							}
							else {

								if (X0.x > 0) {
									new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(P0.x, P0.y));
									new_vg.addLinear(glm::vec2(P0.x, P0.y), glm::vec2(l01.x, l01.y));
								}
								else {
									new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
								}

								//if (X1.x > 0) {
								//	new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(P1.x, P1.y));
								//	new_vg.addLinear(glm::vec2(P1.x, P1.y), glm::vec2(l11.x, l11.y));
								//}
								//else {
								//	new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
								//}

							}
						}
						else if (stroke_line_join == STROKE_JOIN_ROUND) {
							throw std::runtime_error(
								"VGContainer::strokeToOffsetCurve: STROKE_JOIN_ROUNT not implemented.");
							new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
							//new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
						}
						else if (stroke_line_join == STROKE_JOIN_BEVEL) {
							new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
							//new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
						}
						else {
							throw std::runtime_error(
								"VGContainer::strokeToOffsetCurve: illegal stroke line join");
						}
					}

				} // end of curve FOR


				// (4) end cap

				if (contour_colsed) {
					// gen join
					auto vf = end_point.front();
					auto pvf = pos_offset_point.front();
					auto nvf = neg_offset_point.front();
					auto ptf = pos_offset_tan_vector.front();
					auto ntf = neg_offset_tan_vector.front();

					new_vg.addLinear(pos_offset_point.back(), neg_offset_point.back());

					//end_point.push_back(vf);
					//pos_offset_point.push_back(pvf);
					//neg_offset_point.push_back(nvf);
					//pos_offset_tan_vector.push_back(ptf);
					//neg_offset_tan_vector.push_back(ntf);
				}
				else {
					// gen cap

					auto l00 = pos_offset_point.front();
					auto l01 = neg_offset_point.front();

					auto l10 = pos_offset_point.back();
					auto l11 = neg_offset_point.back();

					//new_vg.addLinear(glm::vec2(l01.x, l01.y), glm::vec2(l00.x, l00.y));
					new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));

					/*
					if (stroke_line_cap == STROKE_CAP_BUTT) {

						auto l00 = pos_offset_point.front();
						auto l01 = neg_offset_point.front();

						auto l10 = pos_offset_point.back();
						auto l11 = neg_offset_point.back();

						//new_vg.addLinear(glm::vec2(l01.x, l01.y), glm::vec2(l00.x, l00.y));
						new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
					}
					else if (stroke_line_cap == STROKE_CAP_ROUND) {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: STROKE_CAP_ROUNT not implemented.");
					}
					else if (stroke_line_cap == STROKE_CAP_SQUARE) {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: STROKE_CAP_SQUARE not implemented.");
					}
					else {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: illegal stroke line cap");
					}
					*/
				}

				// (5) neg curves & join

				//for (uint32_t cvi = curve_begin; cvi < curve_end; ++cvi) {
				//for (uint32_t cvi = curve_begin; cvi < 0; ++cvi) {
				for (int32_t cvi = curve_end - 1; cvi >= (int32_t)curve_begin; --cvi) {

					auto curve_reversed = curve.reversed[cvi];
					if (curve_reversed) {
						throw std::runtime_error("DO NOT SUPPORT REVERSED CURVE HERE");
					}

					auto curve_type = curve.type[cvi];
					auto vi = curve.vertexIndex[cvi];

					float2 f2v[4];

					for (int i = 0; i < (curve_type & 7); ++i) {
						auto glmv = vertex.point[vi + i];
						f2v[i] = make_float2(glmv.x, glmv.y);
					}
					if (curve_type == CT_Rational) {
						f2v[3].x = *((float*)&curve.arc_w1s[cvi]);
					}

					SimpleBezierCurve bezier_curve;

					if (curve_type == CT_Linear) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], 0.f);
						new_vg.addLinear(vertex.point[vi + 1], vertex.point[vi], half_stroke_width);
					}
					else if (curve_type == CT_Quadratic) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], 0.f);
						new_vg.addQuadratic(vertex.point[vi + 2], vertex.point[vi + 1],
							vertex.point[vi], half_stroke_width);
					}
					else if (curve_type == CT_Cubic) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2], f2v[3],
							0.f);
						new_vg.addCubic(vertex.point[vi + 3], vertex.point[vi + 2],
							vertex.point[vi + 1], vertex.point[vi], half_stroke_width);
					}
					else if (curve_type == CT_Rational) {
						bezier_curve = SimpleBezierCurve(f2v[0], f2v[1], f2v[2],
							f2v[3].x, 0.f);
						new_vg.addRational(vertex.point[vi + 2], vertex.point[vi + 1],
							vertex.point[vi], *((float*)&curve.arc_w1s[cvi]), half_stroke_width);
					}
					else {
						throw std::runtime_error("VGContainer::strokeToOffsetCurve: illegal curve type");
					}

					// join
					auto i = cvi * 2 - 1;
					if (0 <= i && i + 1< end_point.size()) {

						const auto &l00 = pos_offset_point[i];
						const auto &l01 = pos_offset_point[i + 1];

						const auto &t00 = pos_offset_tan_vector[i];
						const auto &t01 = pos_offset_tan_vector[i + 1];

						const auto &l10 = neg_offset_point[i + 1];
						const auto &l11 = neg_offset_point[i];

						const auto &t10 = neg_offset_tan_vector[i + 1];
						const auto &t11 = neg_offset_tan_vector[i];

						stroke_line_join = STROKE_JOIN_BEVEL;

						if (stroke_line_join == STROKE_JOIN_MITER) {

							// L00 + T00 * a = P0
							// L01 + T01 * b = P0

							//l00.x + t00.x * a = l01.x + t01.x * b;
							//l00.y + t00.y * a = l01.y + t01.y * b;

							//t00.x * a - t01.x * b = l01.x - l00.x;
							//t00.y * a - t01.y * b = l01.y - l00.y;

							glm::mat2x2 A;
							glm::vec2 Y, X0, X1;

							A[0][0] = t00.x;
							A[0][1] = t00.y;

							A[1][0] = t01.x;
							A[1][1] = t01.y;

							Y.x = l01.x - l00.x;
							Y.y = l01.y - l00.y;

							X0 = glm::inverse(A) * Y;

							float2 P0 = l00 + t00 * X0.x;

							// L10 + T10 * c = P1
							// L11 + T11 * d = P!

							// add line : L00-P0, P0-L01
							// add line : L10-P1, P1-L11

							A[0][0] = t10.x;
							A[0][1] = t10.y;

							A[1][0] = t11.x;
							A[1][1] = t11.y;

							Y.x = l11.x - l10.x;
							Y.y = l11.y - l10.y;

							X1 = glm::inverse(A) * Y;

							auto P1 = l10 + t10 * X1.x;

							auto dp = P1 - P0;
							auto d = sqrt(dp.x * dp.x + dp.y * dp.y);

							if (d / stroke_width * 2 > stroke_miter_limit) {
								//new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
								new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
							}
							else {

								//if (X0.x > 0) {
								//	new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(P0.x, P0.y));
								//	new_vg.addLinear(glm::vec2(P0.x, P0.y), glm::vec2(l01.x, l01.y));
								//}
								//else {
								//	new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
								//}

								if (X1.x > 0) {
									new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(P1.x, P1.y));
									new_vg.addLinear(glm::vec2(P1.x, P1.y), glm::vec2(l11.x, l11.y));
								}
								else {
									new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
								}

							}
						}
						else if (stroke_line_join == STROKE_JOIN_ROUND) {
							throw std::runtime_error(
								"VGContainer::strokeToOffsetCurve: STROKE_JOIN_ROUNT not implemented.");
							//new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
							new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
						}
						else if (stroke_line_join == STROKE_JOIN_BEVEL) {
							//new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
							new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
						}
						else {
							throw std::runtime_error(
								"VGContainer::strokeToOffsetCurve: illegal stroke line join");
						}
					}

				} // end of curve FOR

#ifdef HIDE_CAP_AND_JOIN

				if( contour_colsed) {
					// gen join
					auto vf= end_point.front();
					auto pvf = pos_offset_point.front();
					auto nvf = neg_offset_point.front();
					auto ptf = pos_offset_tan_vector.front();
					auto ntf = neg_offset_tan_vector.front();

					end_point.push_back(vf);
					pos_offset_point.push_back(pvf);
					neg_offset_point.push_back(nvf);
					pos_offset_tan_vector.push_back(ptf);
					neg_offset_tan_vector.push_back(ntf);
				}
				else {
					// gen cap
					if (stroke_line_cap == STROKE_CAP_BUTT) {

						auto l00 = pos_offset_point.front();
						auto l01 = neg_offset_point.front();

						auto l10 = pos_offset_point.back();
						auto l11 = neg_offset_point.back();

						new_vg.addLinear(glm::vec2(l01.x, l01.y), glm::vec2(l00.x, l00.y));
						new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
					}
					else if (stroke_line_cap == STROKE_CAP_ROUND) {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: STROKE_CAP_ROUNT not implemented.");
					}
					else if (stroke_line_cap == STROKE_CAP_SQUARE) {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: STROKE_CAP_SQUARE not implemented.");
					}
					else {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: illegal stroke line cap");
					}					
				}

				// join
				for (int i = 1; i + 1 < end_point.size(); i+=2) {

					const auto &l00 = pos_offset_point[i];
					const auto &l01 = pos_offset_point[i + 1];

					const auto &t00 = pos_offset_tan_vector[i];
					const auto &t01 = pos_offset_tan_vector[i + 1];

					const auto &l10 = neg_offset_point[i + 1];
					const auto &l11 = neg_offset_point[i];

					const auto &t10 = neg_offset_tan_vector[i + 1];
					const auto &t11 = neg_offset_tan_vector[i];

					if (stroke_line_join == STROKE_JOIN_MITER) {

						// L00 + T00 * a = P0
						// L01 + T01 * b = P0

						//l00.x + t00.x * a = l01.x + t01.x * b;
						//l00.y + t00.y * a = l01.y + t01.y * b;

						//t00.x * a - t01.x * b = l01.x - l00.x;
						//t00.y * a - t01.y * b = l01.y - l00.y;

						glm::mat2x2 A;
						glm::vec2 Y, X0, X1;

						A[0][0] = t00.x;
						A[0][1] = t00.y;

						A[1][0] = t01.x;
						A[1][1] = t01.y;

						Y.x = l01.x - l00.x;
						Y.y = l01.y - l00.y;

						X0 = glm::inverse(A) * Y;

						float2 P0 = l00 + t00 * X0.x;

						// L10 + T10 * c = P1
						// L11 + T11 * d = P!

						// add line : L00-P0, P0-L01
						// add line : L10-P1, P1-L11

						A[0][0] = t10.x;
						A[0][1] = t10.y;

						A[1][0] = t11.x;
						A[1][1] = t11.y;

						Y.x = l11.x - l10.x;
						Y.y = l11.y - l10.y;

						X1 = glm::inverse(A) * Y;

						auto P1 = l10 + t10 * X1.x;

						auto dp = P1 - P0;
						auto d = sqrt(dp.x * dp.x + dp.y * dp.y);

						if (d / stroke_width * 2 > stroke_miter_limit) {
							new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
							new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
						}
						else {

							if (X0.x > 0) {
								new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(P0.x, P0.y));
								new_vg.addLinear(glm::vec2(P0.x, P0.y), glm::vec2(l01.x, l01.y));
							}
							else {
								new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
							}

							if (X1.x > 0) {
								new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(P1.x, P1.y));
								new_vg.addLinear(glm::vec2(P1.x, P1.y), glm::vec2(l11.x, l11.y));
							}
							else {
								new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
							}

						}
					} 
					else if (stroke_line_join == STROKE_JOIN_ROUND) {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: STROKE_JOIN_ROUNT not implemented.");
						new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
						new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
					}
					else if (stroke_line_join == STROKE_JOIN_BEVEL) {
						new_vg.addLinear(glm::vec2(l00.x, l00.y), glm::vec2(l01.x, l01.y));
						new_vg.addLinear(glm::vec2(l10.x, l10.y), glm::vec2(l11.x, l11.y));
					}
					else {
						throw std::runtime_error(
							"VGContainer::strokeToOffsetCurve: illegal stroke line join");
					}

				}
#endif

			} // end of contour FOR
		} // end of stroke IF
	} // end of path FOR

	return new_vg;
}

// -------- -------- -------- -------- -------- -------- -------- --------
VGContainer VGContainer::splitStrokeAndFill() {

	VGContainer new_vg;

	new_vg.x = x;
	new_vg.y = y;
	new_vg.width = width;
	new_vg.height = height;
	new_vg.vp_pos_x = vp_pos_x;
	new_vg.vp_pos_y = vp_pos_y;

	for (uint32_t pi = 0; pi < pathNumber(); ++pi) {

		auto clone_path = [&](bool close_contour) {
			auto contour_begin = path.contourIndex[pi];
			auto contour_n = path.contourNumber[pi];
			auto contour_end = contour_begin + contour_n;

			for (uint32_t cti = contour_begin; cti < contour_end; ++cti) {

				auto new_contour_id = new_vg.addContour();

				auto contour_closed = this->contour.closed[cti];
				new_vg.contour.closed[new_contour_id] = contour_closed;

				auto curve_begin = contour.curveIndex[cti];
				auto curve_n = contour.curveNumber[cti];
				auto curve_end = curve_begin + curve_n;

				uint32_t contour_first_vertex_i = 0;
				uint32_t contour_last_vertex_i = 0;

				float2 contour_first_vertex;
				float2 contour_last_vertex;

				for (uint32_t cvi = curve_begin; cvi < curve_end; ++cvi) {

					auto curve_type = curve.type[cvi];
					auto vi = curve.vertexIndex[cvi];
					auto curve_reversed = curve.reversed[cvi];
					auto curve_offset = curve.offset[cvi];
					auto curve_w = *((float*)&curve.arc_w1s[cvi]);

					if (curve_reversed) {
						throw std::runtime_error("DO NOT SUPPORT REVERSED CURVE HERE");
					}

					if (cvi == curve_begin) { contour_first_vertex_i = vi; }
					contour_last_vertex_i = vi + (curve_type & 7) - 1;

					bool skip_curve = false;

					SimpleBezierCurve bezier_curve;
					if (curve_type == CT_Linear) {
						bezier_curve = SimpleBezierCurve(
							make_float2(vertex.point[vi].x, vertex.point[vi].y),
							make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
							curve_offset
							);
						if (bezier_curve[0] == bezier_curve[1]) {
							skip_curve = true;
						}
					}
					else if (curve_type == CT_Quadratic) {
						bezier_curve = SimpleBezierCurve(
							make_float2(vertex.point[vi].x, vertex.point[vi].y),
							make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
							make_float2(vertex.point[vi + 2].x, vertex.point[vi + 2].y),
							curve_offset
							);
						if (bezier_curve[0] == bezier_curve[1]
							&& bezier_curve[1] == bezier_curve[2]
							) {
							skip_curve = true;
						}
					}
					else if (curve_type == CT_Cubic) {
						bezier_curve = SimpleBezierCurve(
							make_float2(vertex.point[vi].x, vertex.point[vi].y),
							make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
							make_float2(vertex.point[vi + 2].x, vertex.point[vi + 2].y),
							make_float2(vertex.point[vi + 3].x, vertex.point[vi + 3].y),
							curve_offset
							);
						if (bezier_curve[0] == bezier_curve[1]
							&& bezier_curve[1] == bezier_curve[2]
							&& bezier_curve[2] == bezier_curve[3]
							) {
							skip_curve = true;
						}
					}
					else if (curve_type == CT_Rational) {
						bezier_curve = SimpleBezierCurve(
							make_float2(vertex.point[vi].x, vertex.point[vi].y),
							make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
							make_float2(vertex.point[vi + 2].x, vertex.point[vi + 2].y),
							curve_w,
							curve_offset
							);
						if (bezier_curve[0] == bezier_curve[1]
							&& bezier_curve[1] == bezier_curve[2]
							) {
							skip_curve = true;
						}
					}
					else {
						throw std::runtime_error("VGContainer::strokeToFill: illegal curve type");
					}

					if (!skip_curve) { new_vg.addCurve(bezier_curve); }

					if (cvi == curve_begin) {
						contour_first_vertex = bezier_curve.front();
					}

					if (cvi + 1 == curve_end) {
						contour_last_vertex = bezier_curve.back();
					}

				} // end of curve loop

				if (close_contour && contour_first_vertex != contour_last_vertex) {
					new_vg.addLinear(contour_first_vertex, contour_last_vertex);
				}

			} // end of contour loop
		};

		// fill path
		auto fill_type = path.fillType[pi];
		auto stroke_type = path.strokeType[pi];

		if (fill_type != FT_NONE) {

			auto npi = new_vg.addPath();

			new_vg.path.fillType[npi] = path.fillType[pi];
			new_vg.path.fillColor[npi] = path.fillColor[pi];
			new_vg.path.fillIndex[npi] = path.fillIndex[pi];
			new_vg.path.fillOpacity[npi] = path.fillOpacity[pi];
			new_vg.path.fillRule[npi] = path.fillRule[pi];

			new_vg.path.strokeType[npi] = ST_NONE;

			clone_path(true);
		}

		if (stroke_type != ST_NONE) {

			auto npi = new_vg.addPath();

			new_vg.path.fillType[npi] = FT_NONE;

			new_vg.path.strokeType[npi] = path.strokeType[pi];
			new_vg.path.strokeColor[npi] = path.strokeColor[pi];
			new_vg.path.strokeIndex[npi] = path.strokeIndex[pi];
			new_vg.path.strokeOpacity[npi] = path.strokeOpacity[pi];

			new_vg.path.strokeWidth[npi] = path.strokeWidth[pi];
			new_vg.path.strokeLineCap[npi] = path.strokeLineCap[pi];
			new_vg.path.strokeLineJoin[npi] = path.strokeLineJoin[pi];
			new_vg.path.strokeMiterLimit[npi] = path.strokeMiterLimit[pi];

			clone_path(false);
		}

	} // end of path FOR

	return new_vg;
}

// -------- -------- -------- -------- -------- -------- -------- --------
VGContainer VGContainer::subdiv() {

	VGContainer new_vg;

	new_vg.x = x;
	new_vg.y = y;
	new_vg.width = width;
	new_vg.height = height;
	new_vg.vp_pos_x = vp_pos_x;
	new_vg.vp_pos_y = vp_pos_y;

	for (uint32_t pi = 0; pi < pathNumber(); ++pi) {

		// fill path
		auto ft = path.fillType[pi];

		auto npi = new_vg.addPath();

		new_vg.path.fillType[npi] = path.fillType[pi];
		new_vg.path.fillColor[npi] = path.fillColor[pi];
		new_vg.path.fillIndex[npi] = path.fillIndex[pi];
		new_vg.path.fillOpacity[npi] = path.fillOpacity[pi];
		new_vg.path.fillRule[npi] = path.fillRule[pi];

		new_vg.path.strokeType[npi] = ST_NONE;

		auto contour_begin = path.contourIndex[pi];
		auto contour_n = path.contourNumber[pi];
		auto contour_end = contour_begin + contour_n;

		for (uint32_t cti = contour_begin; cti < contour_end; ++cti) {

			new_vg.addContour();

			auto curve_begin = contour.curveIndex[cti];
			auto curve_n = contour.curveNumber[cti];
			auto curve_end = curve_begin + curve_n;

			uint32_t contour_first_vertex_i = 0;
			uint32_t contour_last_vertex_i = 0;

			for (uint32_t cvi = curve_begin; cvi < curve_end; ++cvi) {

				auto curve_type = curve.type[cvi];
				auto vi = curve.vertexIndex[cvi];
				auto curve_reversed = curve.reversed[cvi];
				auto curve_offset = curve.offset[cvi];
				auto curve_w = *((float*)&curve.arc_w1s[cvi]);

				if (curve_reversed) {
					throw std::runtime_error("DO NOT SUPPORT REVERSED CURVE HERE");
				}

				if (cvi == curve_begin) { contour_first_vertex_i = vi; }
				contour_last_vertex_i = vi + (curve_type & 7) - 1;

				SimpleBezierCurve bezier_curve;
				if (curve_type == CT_Linear) {
					bezier_curve = SimpleBezierCurve(
						make_float2(vertex.point[vi].x, vertex.point[vi].y),
						make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
						curve_offset
						);
				}
				else if (curve_type == CT_Quadratic) {
					bezier_curve = SimpleBezierCurve(
						make_float2(vertex.point[vi].x, vertex.point[vi].y),
						make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
						make_float2(vertex.point[vi + 2].x, vertex.point[vi + 2].y),
						curve_offset
						);
				}
				else if (curve_type == CT_Cubic) {
					bezier_curve = SimpleBezierCurve(
						make_float2(vertex.point[vi].x, vertex.point[vi].y),
						make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
						make_float2(vertex.point[vi + 2].x, vertex.point[vi + 2].y),
						make_float2(vertex.point[vi + 3].x, vertex.point[vi + 3].y),
						curve_offset
						);
				}
				else if (curve_type == CT_Rational) {
					bezier_curve = SimpleBezierCurve(
						make_float2(vertex.point[vi].x, vertex.point[vi].y),
						make_float2(vertex.point[vi + 1].x, vertex.point[vi + 1].y),
						make_float2(vertex.point[vi + 2].x, vertex.point[vi + 2].y),
						curve_w,
						curve_offset
						);
				}
				else {
					throw std::runtime_error("VGContainer::strokeToFill: illegal curve type");
				}

				// subdiv.

				std::function<void(float, float, int)> bc_subdiv;

				bc_subdiv = [&](float left, float right, int depth) {

					auto p0 = bezier_curve.point(left);
					auto p1 = bezier_curve.point(right);
					auto m = (left + right) * .5f;
					auto curve_m = bezier_curve.point(m);
					auto line_m = (p0 + p1)  * .5f;

					if (depth > 64) {
						new_vg.addLinear(p0, p1);
						return;
					}

					if (length(line_m - curve_m) < 1 / 64.f) {
						new_vg.addLinear(p0, p1);
						return;
					}

					bc_subdiv(left, m, depth+1);
					bc_subdiv(m, right, depth+1);

				};

				if (curve_type == CT_Linear) {
					new_vg.addLinear(
						bezier_curve.point(0.f),
						bezier_curve.point(1.f)
						);
				}
				else {
					bc_subdiv(0.f, 1.f, 1);
				}

			}

		}

	} // end of path FOR

	return new_vg;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGContainer::save(const std::string &i_filename) const {

	//if (Config::Verbose()) {
	//	printf("VGContainer::save: \"%s\"\n", i_filename.c_str());
	//}

	stdext::string filename = i_filename;
	auto suffix = filename.right(3);
	if (suffix == "vgt") { saveTxt(i_filename); }
	if (suffix == "vgb") { saveBinary(i_filename); }
	else { printf("VGContainer::save\n"); }
}

void VGContainer::load(const std::string &i_filename) {

	//if (Config::Verbose()) {
	//	printf("VGContainer::load: \"%s\"\n", i_filename.c_str());
	//}

	stdext::string filename = i_filename;
	auto suffix = filename.right(3);
	if (suffix == "vgt") { loadTxt(i_filename); }
	if (suffix == "vgb") { loadBinary(i_filename); }
	else { printf("VGContainer::load\n"); }
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGContainer::saveTxt(const std::string &filename) const {
	FILE *fout = fopen(filename.c_str(), "wb");
	if (!fout) {
		printf("VGContainer::save: cannot open file %s", filename.c_str());
	}

	uint32_t n_path = pathNumber();
	uint32_t n_contour = contourNumber();
	uint32_t n_curve = curveNumber();
	uint32_t n_vertex = vertexNumber();
	uint32_t size_gradient_ramp = (uint32_t)gradient.gradient_ramp_texture.size();
	uint32_t size_gradient_table = (uint32_t)gradient.gradient_table_texture.size();

	fprintf(fout, "%d,%d,%d,%d\n", x, y, width, height);

	fprintf(fout, "%u,%u,%u,%u,%u,%u\n",
		n_path, n_contour, n_curve, n_vertex, size_gradient_ramp, size_gradient_table);

	for (uint32_t i = 0; i < n_path; ++i) {
		fprintf(fout, "%u,%u,%c,%c,%u,%u\n",
			path.contourIndex[i],
			path.contourNumber[i],
			path.fillType[i],
			path.fillRule[i],
			(uint32_t)path.fillColor[i],
			path.fillIndex[i]
			);
	}
	for (uint32_t i = 0; i < n_contour; ++i) {
		fprintf(fout, "%u,%u\n", contour.curveIndex[i], contour.curveNumber[i]);
	}
	for (uint32_t i = 0; i < n_curve; ++i) {
		fprintf(fout, "%u,%c,%d\n",
			curve.vertexIndex[i], curve.type[i], curve.arc_w1s[i]);
	}
	for (uint32_t i = 0; i < n_vertex; ++i) {
		fprintf(fout, "%e,%e\n", vertex.point[i].x, vertex.point[i].y);
	}
	for (uint32_t i = 0; i < size_gradient_ramp; ++i) {
		fprintf(fout, "%d\n", gradient.gradient_ramp_texture[i]);
	}
	for (uint32_t i = 0; i < size_gradient_table; ++i) {
		fprintf(fout, "%d\n", gradient.gradient_table_texture[i]);
	}

	fclose(fout);
}

void VGContainer::loadTxt(const std::string &filename) {
	FILE *fin = fopen(filename.c_str(), "rb");
	if (!fin) {
		printf("VGContainer::save: cannot open file %s", filename.c_str());
	}

	uint32_t n_path;
	uint32_t n_contour;
	uint32_t n_curve;
	uint32_t n_vertex;
	uint32_t size_gradient_ramp;
	uint32_t size_gradient_table;

	fscanf(fin, "%d,%d,%d,%d\n", &x, &y, &width, &height);

	fscanf(fin, "%u,%u,%u,%u,%u,%u",
		&n_path, &n_contour, &n_curve, &n_vertex, &size_gradient_ramp, &size_gradient_table);

	path.resize(n_path);
	contour.resize(n_contour);
	curve.resize(n_curve);
	vertex.resize(n_vertex);
	gradient.resize(size_gradient_ramp, size_gradient_table);

	for (uint32_t i = 0; i < n_path; ++i) {
		fscanf(fin, "%u,%u,%c,%c,%u,%u",
			&path.contourIndex[i],
			&path.contourNumber[i],
			&path.fillType[i],
			&path.fillRule[i],
			(uint32_t*)&path.fillColor[i],
			&path.fillIndex[i]
			);
	}
	for (uint32_t i = 0; i < n_contour; ++i) {
		fscanf(fin, "%u,%u\n", &contour.curveIndex[i], &contour.curveNumber[i]);
	}
	for (uint32_t i = 0; i < n_curve; ++i) {
		fscanf(fin, "%u,%c,%d\n",
			&curve.vertexIndex[i], &curve.type[i], &curve.arc_w1s[i]);
	}
	for (uint32_t i = 0; i < n_vertex; ++i) {
		fscanf(fin, "%e,%e\n", &vertex.point[i].x, &vertex.point[i].y);
	}
	for (uint32_t i = 0; i < size_gradient_ramp; ++i) {
		fscanf(fin, "%d\n", &gradient.gradient_ramp_texture[i]);
	}
	for (uint32_t i = 0; i < size_gradient_table; ++i) {
		fscanf(fin, "%d\n", &gradient.gradient_table_texture[i]);
	}

	fclose(fin);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void VGContainer::saveBinary(const std::string &filename) const {
	FILE *fout = fopen(filename.c_str(), "wb");
	if (!fout) {
		printf("VGContainer::save: cannot open file %s", filename.c_str());
	}

	uint32_t n_path = pathNumber();
	uint32_t n_contour = contourNumber();
	uint32_t n_curve = curveNumber();
	uint32_t n_vertex = vertexNumber();
	uint32_t size_gradient_ramp = (uint32_t)gradient.gradient_ramp_texture.size();
	uint32_t size_gradient_table = (uint32_t)gradient.gradient_table_texture.size();

	fwrite(&x, sizeof(uint32_t), 1, fout);
	fwrite(&y, sizeof(uint32_t), 1, fout);
	fwrite(&width, sizeof(uint32_t), 1, fout);
	fwrite(&height, sizeof(uint32_t), 1, fout);

	fwrite(&n_path, sizeof(uint32_t), 1, fout);
	fwrite(&n_contour, sizeof(uint32_t), 1, fout);
	fwrite(&n_curve, sizeof(uint32_t), 1, fout);
	fwrite(&n_vertex, sizeof(uint32_t), 1, fout);
	fwrite(&size_gradient_ramp, sizeof(uint32_t), 1, fout);
	fwrite(&size_gradient_table, sizeof(uint32_t), 1, fout);

	//
	fwrite(path.contourIndex.data(), sizeof(uint32_t), n_path, fout);
	fwrite(path.contourNumber.data(), sizeof(uint32_t), n_path, fout);

	fwrite(path.fillType.data(), sizeof(uint8_t), n_path, fout);
	fwrite(path.fillRule.data(), sizeof(uint8_t), n_path, fout);
	fwrite(path.fillColor.data(), sizeof(u8rgba), n_path, fout);
	fwrite(path.fillIndex.data(), sizeof(uint32_t), n_path, fout);

	//
	fwrite(contour.curveIndex.data(), sizeof(uint32_t), n_contour, fout);
	fwrite(contour.curveNumber.data(), sizeof(uint32_t), n_contour, fout);

	//
	fwrite(curve.vertexIndex.data(), sizeof(uint32_t), n_curve, fout);
	fwrite(curve.type.data(), sizeof(uint8_t), n_curve, fout);
	//fwrite(curve.type.data(), sizeof(uint32_t), n_curve, fout);
	fwrite(curve.arc_w1s.data(), sizeof(int), n_curve, fout);

	//
	fwrite(vertex.point.data(), sizeof(float), n_vertex * 2, fout);

	//
	fwrite(gradient.gradient_ramp_texture.data(), sizeof(int), size_gradient_ramp, fout);
	fwrite(gradient.gradient_table_texture.data(), sizeof(int), size_gradient_table, fout);

	fclose(fout);
}

void VGContainer::loadBinary(const std::string &filename) {
	FILE *fin = fopen(filename.c_str(), "rb");
	if (!fin) {
		printf("VGContainer::save: cannot open file %s", filename.c_str());
	}

	uint32_t n_path;
	uint32_t n_contour;
	uint32_t n_curve;
	uint32_t n_vertex;
	uint32_t size_gradient_ramp;
	uint32_t size_gradient_table;

	fread(&x, sizeof(uint32_t), 1, fin);
	fread(&y, sizeof(uint32_t), 1, fin);
	fread(&width, sizeof(uint32_t), 1, fin);
	fread(&height, sizeof(uint32_t), 1, fin);

	fread(&n_path, sizeof(uint32_t), 1, fin);
	fread(&n_contour, sizeof(uint32_t), 1, fin);
	fread(&n_curve, sizeof(uint32_t), 1, fin);
	fread(&n_vertex, sizeof(uint32_t), 1, fin);
	fread(&size_gradient_ramp, sizeof(uint32_t), 1, fin);
	fread(&size_gradient_table, sizeof(uint32_t), 1, fin);

	//
	path.resize(n_path);
	contour.resize(n_contour);
	curve.resize(n_curve);
	vertex.resize(n_vertex);
	gradient.resize(size_gradient_ramp, size_gradient_table);

	//
	fread(path.contourIndex.data(), sizeof(uint32_t), n_path, fin);
	fread(path.contourNumber.data(), sizeof(uint32_t), n_path, fin);

	fread(path.fillType.data(), sizeof(uint8_t), n_path, fin);
	fread(path.fillRule.data(), sizeof(uint8_t), n_path, fin);
	fread(path.fillColor.data(), sizeof(u8rgba), n_path, fin);
	fread(path.fillIndex.data(), sizeof(uint32_t), n_path, fin);

	//
	fread(contour.curveIndex.data(), sizeof(uint32_t), n_contour, fin);
	fread(contour.curveNumber.data(), sizeof(uint32_t), n_contour, fin);

	//
	fread(curve.vertexIndex.data(), sizeof(uint32_t), n_curve, fin);
	fread(curve.type.data(), sizeof(uint8_t), n_curve, fin);
	//fread(curve.type.data(), sizeof(uint32_t), n_curve, fin);
	fread(curve.arc_w1s.data(), sizeof(int), n_curve, fin);

	//
	fread(vertex.point.data(), sizeof(float), n_vertex * 2, fin);

	//
	fread(gradient.gradient_ramp_texture.data(), sizeof(int), size_gradient_ramp, fin);
	fread(gradient.gradient_table_texture.data(), sizeof(int), size_gradient_table, fin);

	fclose(fin);
}

void VGContainer::saveLuaRvg(const std::string &fileName) const {
	FILE *fout = fopen(fileName.c_str(), "w");
	if (!fout) {
		printf("VGContainer::saveLuaRvg: cannot open ouput file %s\n", fileName.c_str());
	}

	fprintf(fout, "local _M = {}\n\n");

	fprintf(fout, "function _M.window(drv)\n");
	fprintf(fout, "    return drv.window(0, 0, %d, %d);\n", width, height);
	fprintf(fout, "end\n\n");

	fprintf(fout, "function _M.viewport(drv)\n");
	fprintf(fout, "    return drv.viewport(0, 0, %d, %d);\n", width, height);
	fprintf(fout, "end\n\n");

	fprintf(fout,
		R"(function _M.scene(drv)
  local M = drv.command.move_to_abs
  local L = drv.command.line_to_abs
  local Q = drv.command.quad_to_abs
  local C = drv.command.cubic_to_abs
  local Z = drv.command.close_path_abs
  local mod = drv.spread.mod
  local clamp = drv.spread.clamp
  local mirror = drv.spread.mirror
  local p2 = drv.p2
  local rgb = drv.rgb
  local solid = drv.solid
  local color = function (r,g,b,a,o) return solid(rgb(r,g,b,a),o) end
  local path = drv.path
  local eofill = drv.eofill
  local nzfill = drv.fill
  local ramp = drv.ramp
  local affine = drv.affine
  local lineargradient = drv.lineargradient
  local radialgradient = drv.radialgradient
  local a,b,w
  local s = {}
)");

	// -------- -------- -------- --------
	// pathes

	//a = path{M,154.253,73.865,C,154.253,72.133,152.847,70.727,151.115,70.727,149.383,70.727,147.977,72.133,147.977,73.865,147.977,75.597,149.383,77.003,151.115,77.003,152.847,77.003,154.253,75.597,154.253,73.865,}
	//b = color(1,1,1,1,1)
	//s[204] = eofill(a, b)

	std::string d;

	const auto &vg = *this;

	auto v2s = [](const glm::vec2 &p) {
		char s[64] = { 0 };
		sprintf(s, "%f,%f,", p.x, p.y);
		return std::string(s);
	};

	//
	for (uint32_t i = 0; i < vg.pathNumber(); ++i) {
		std::string fill_rule_str = vg.path.fillRule[i] ? "evenodd" : "nonzero";
		auto fill_color = vg.path.fillColor[i];
		//char fill_str[32];
		float fill_opacity = 1.f;

		fprintf(fout, "\n");
		if (vg.path.fillType[i] == FT_COLOR) {
			fprintf(fout, "  b = color(%f,%f,%f,%f,1)\n",
				fill_color.r / 255.f, fill_color.g / 255.f, fill_color.b / 255.f,
				fill_color.a / 255.f);
		}
		else {
			//auto gid = vg.path.fillIndex[i];
			//sprintf(fill_str, "url(#G%d)", gid);
			//fill_opacity = vg.path.fillOpacity[i];
			//uint8_t u8_opacity = fill_opacity * 255.f;
			//fill_opacity = u8_opacity / 255.f;
		}

		auto contour_pos = vg.path.contourIndex[i];
		auto contour_number = vg.path.contourNumber[i];

		d = "";

		//Bitmap bmp;

		//bmp.resize(1024, 1024);
		//bmp.fill(u8rgba(0, 0, 0, 255));

		for (uint32_t j = 0; j < contour_number; ++j) {
			auto curve_pos = vg.contour.curveIndex[contour_pos + j];
			auto curve_number = vg.contour.curveNumber[contour_pos + j];

			for (uint32_t k = 0; k < curve_number; ++k) {
				auto vertex_pos = vg.curve.vertexIndex[curve_pos + k];
				auto curve_type = vg.curve.type[curve_pos + k];

				if (k == 0) {
					d.push_back('M');
					d.push_back(',');
					d += v2s(vg.vertex.point[vertex_pos]);
				}

				if (curve_type == CT_Linear) { d.push_back('L'); }
				else if (curve_type == CT_Quadratic) { d.push_back('Q'); }
				else if (curve_type == CT_Cubic) { d.push_back('C'); }
				else if (curve_type == CT_Rational) {

					char rstr[128];
					auto w = *((float*)&vg.curve.arc_w1s[curve_pos + k]);
					sprintf(rstr, " A,%f,%f,%f,%f,%f,",
						vg.vertex.point[vertex_pos + 1].x * w,
						vg.vertex.point[vertex_pos + 1].y * w,
						w,
						vg.vertex.point[vertex_pos + 2].x,
						vg.vertex.point[vertex_pos + 2].y
						);

					d.push_back(',');
					d = d + rstr;
				}
				else { throw std::runtime_error("SVG::save illegal curve type."); }

				d.push_back(',');
				if (curve_type != CT_Rational) {
					for (int l = 1; l < (curve_type & 7); ++l) {
						d += v2s(vg.vertex.point[vertex_pos + l]);
					}
				}

			}
		}

		d.push_back('Z');
		d.push_back(',');

		for (;;) {
			auto i = d.find(",,");
			if (i == std::string::npos) {
				break;
			}
			d.erase(d.begin() + i);
		}

		//a = path{M,154.253,73.865,C,154.253,72.133,152.847,70.727,151.115,70.727,149.383,70.727,147.977,72.133,147.977,73.865,147.977,75.597,149.383,77.003,151.115,77.003,152.847,77.003,154.253,75.597,154.253,73.865,}
		//b = color(1,1,1,1,1)
		//s[204] = eofill(a, b)

		fprintf(fout, "  a = path{%s}\n", d.c_str());

		if (vg.path.fillRule[i] == 0) {
			fprintf(fout, "  s[%d] = nzfill(a, b)\n", i + 1);
		}
		else {
			fprintf(fout, "  s[%d] = eofill(a, b)\n", i + 1);
		}
	}


	// end of pathes
	// -------- -------- -------- --------

	fprintf(fout, "\n  return drv.scene(s)\n");
	fprintf(fout, "end\n\n");
	fprintf(fout, "return _M\n");

	fclose(fout);
}

//} // end of namespace OldVGContainer

namespace NewVGContainer  {

} // end of namespace NewVGContainer

} // end of namespace Mochimazui
