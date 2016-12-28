
#ifndef _MOCHIMAZUI_VG_CONTAINER_H_
#define _MOCHIMAZUI_VG_CONTAINER_H_

#include <map>
#include <vector>
#include <exception>
#include <stdexcept>

#include <glm/glm.hpp>

#include <mochimazui/color.h>

#include "gradient.h"
#include "bezier_curve.h"

namespace Mochimazui {

enum VGFillType {
	FT_NONE = 0,
	FT_COLOR = 1,
	FT_GRADIENT = 4
};

enum VGStrokeType {
	ST_NONE = 0,
	ST_COLOR = 1,
	ST_GRADIENT = 4
};

enum VGStrokeCapType {
	STROKE_CAP_BUTT = 0,
	STROKE_CAP_ROUND = 1,
	STROKE_CAP_SQUARE = 2
};

enum VGStrokeJoinType {
	STROKE_JOIN_MITER = 0,
	STROKE_JOIN_ROUND = 1,
	STROKE_JOIN_BEVEL = 2
};

enum VGFillRule {
	FR_NON_ZERO = 0,
	FR_EVEN_ODD = 1
};

enum VGArcSize {
	ARC_SIZE_LARGE = 1,
	ARC_SIZE_SMALL = 0
};

enum VGArcSweep {
	ARC_SWEEP_POSITIVE = 1,
	ARC_SWEEP_NEGATIVE = 0
};

//the memory layout is significant when packing textures in rvg.cpp, DO NOT TOUCH
struct TGradientItem {
	float tr[6];
	int p, n;
	float focal_point[4];
	int mode;
};

class VGContainerCannotOpenFile : public std::exception {
};

class VGContainer {

public:
	static uint8_t curveVertexNumber(uint8_t ct) { return ct & 0x0F; }

public:
	uint32_t vertexNumber() const { return (uint32_t)vertex.point.size(); }
	uint32_t curveNumber()  const { return (uint32_t)curve.vertexIndex.size(); }
	uint32_t contourNumber()  const { return (uint32_t)contour.curveIndex.size(); }
	uint32_t pathNumber()  const { return (uint32_t)path.contourIndex.size(); }

public:
	uint32_t addVertex(int number = 1);
	uint32_t addVertex(const glm::vec2 &v);
	uint32_t addVertex(const glm::vec2 &v0, const glm::vec2 &v1);
	uint32_t addVertex(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2);
	uint32_t addVertex(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3);

	// curve without offset
	uint32_t addCurve(VGCurveType tc);
	uint32_t addLinear(const glm::vec2 &v0, const glm::vec2 &v1);
	uint32_t addQuadratic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2);
	uint32_t addCubic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3);
	uint32_t addRational(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float w);

	uint32_t addLinear(const float2 &v0, const float2 &v1, float offset = 0.f);
	uint32_t addQuadratic(const float2 &v0, const float2 &v1, const float2 &v2, float offset = 0.f);
	uint32_t addCubic(const float2 &v0, const float2 &v1, const float2 &v2, const float2 &v3, float offset = 0.f);
	uint32_t addRational(const float2 &v0, const float2 &v1, const float2 &v2, float w, float offset = 0.f);

	// curve with offset
	uint32_t addCurve(VGCurveType tc, float offset);
	uint32_t addLinear(const glm::vec2 &v0, const glm::vec2 &v1, float offset);
	uint32_t addQuadratic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float offset);
	uint32_t addCubic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3, float offset);
	uint32_t addRational(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float w, float offset);

	uint32_t addCurve(const SimpleBezierCurve &bc);

	//
	void addSVGArc(glm::vec2 vstart, float rx, float ry,
		float x_axis_rotation, int larg_flag, int sweep_flag, glm::vec2 v_end);

	//
	uint32_t addContour();
	uint32_t addPath();
	uint32_t addPath(int f_rule, int f_mode, u8rgba f_color, int f_index);
	uint32_t addStrokePath(u8rgba f_color, float width = 1.f);

public:
	void reserve(int path, int contour, int curve, int vertex);
	void clear();

public:
	VGContainer strokeToOffsetCurve();
	VGContainer splitStrokeAndFill();

	VGContainer subdiv();

	void cutToLength();

	void reduceDegenerate();
	void mergeAdjacentPath();

public:
	void save(const std::string &) const;
	void load(const std::string &);

	void saveLuaRvg(const std::string &) const;

private:
	void saveTxt(const std::string &) const;
	void loadTxt(const std::string &);

	void saveBinary(const std::string &) const;
	void loadBinary(const std::string &);

public:

	struct VertexData {
		std::vector<glm::vec2> point;
		//std::vector<float2> point;

		void reserve(size_t s) { point.reserve(s); }
		void resize(size_t s) { point.resize(s); }
		void clear() { point.clear(); }
	};

	struct CurveData {
		std::vector<uint32_t> vertexIndex;
		std::vector<uint8_t> type;
		std::vector<uint8_t> reversed;
		std::vector<int> arc_w1s;
		std::vector<float> offset;

		void reserve(size_t s) {
			vertexIndex.reserve(s);
			type.reserve(s);
			reversed.reserve(s);
			arc_w1s.reserve(s);
			offset.reserve(s);
		}

		void resize(size_t s) {
			vertexIndex.resize(s);
			type.resize(s);
			reversed.resize(s);
			arc_w1s.resize(s);
			offset.resize(s);
		}

		void clear() {
			vertexIndex.clear();
			type.clear();
			reversed.clear();
			arc_w1s.clear();
			offset.clear();
		}
	};

	struct ContourData {
		std::vector<uint32_t> curveIndex;
		std::vector<uint32_t> curveNumber;
		std::vector<uint8_t> closed;

		void reserve(size_t s) {
			curveIndex.reserve(s);
			curveNumber.reserve(s);
			closed.reserve(s);
		}

		void resize(size_t s) {
			curveIndex.resize(s);
			curveNumber.resize(s);
			closed.resize(s);
		}

		void clear() {
			curveIndex.clear();
			curveNumber.clear();
			closed.clear();
		}
	};

	struct PathData {

	public:
		// contour info
		std::vector<uint32_t> contourIndex;
		std::vector<uint32_t> contourNumber;

		// fill 
		std::vector<uint8_t> fillType;
		std::vector<u8rgba> fillColor;
		std::vector<uint32_t> fillIndex;
		std::vector<float> fillOpacity;

		// fill-ext
		std::vector<uint8_t> fillRule;

		// stroke
		std::vector<uint8_t> strokeType;
		std::vector<u8rgba> strokeColor;
		std::vector<uint32_t> strokeIndex;
		std::vector<float> strokeOpacity;

		// stroke-ext
		std::vector<float> strokeWidth;
		std::vector<uint8_t> strokeLineCap;
		std::vector<uint8_t> strokeLineJoin;
		std::vector<float> strokeMiterLimit;
		//std::vector<std::string> strokeDathArray;
		//std::vector<float> strokeDashOffset;

		// gradient
		std::vector<std::string> gradientHref;

		// raw path string
		std::vector<std::string> svgString;
		std::vector<std::vector<uint8_t>> nvprPathCommands;
		std::vector<std::vector<float>> nvprPathCoords;

	public:

		void reserve(size_t s) {

			contourIndex.reserve(s);
			contourNumber.reserve(s);

			fillType.reserve(s);
			fillColor.reserve(s);
			fillIndex.reserve(s);
			fillOpacity.reserve(s);

			fillRule.reserve(s);

			strokeType.reserve(s);
			strokeColor.reserve(s);
			strokeIndex.reserve(s);
			strokeOpacity.reserve(s);

			strokeWidth.reserve(s);
			strokeLineCap.reserve(s);
			strokeLineJoin.reserve(s);
			strokeMiterLimit.reserve(s);

			gradientHref.reserve(s);

			svgString.reserve(s);
			nvprPathCommands.reserve(s);
			nvprPathCoords.reserve(s);
		}

		void resize(size_t s) {

			contourIndex.resize(s);
			contourNumber.resize(s);

			fillType.resize(s);
			fillColor.resize(s);
			fillIndex.resize(s);
			fillOpacity.resize(s);

			fillRule.resize(s);

			strokeType.resize(s);
			strokeColor.resize(s);
			strokeIndex.resize(s);
			strokeOpacity.resize(s);

			strokeWidth.resize(s);
			strokeLineCap.resize(s);
			strokeLineJoin.resize(s);
			strokeMiterLimit.resize(s);

			gradientHref.resize(s);

			svgString.resize(s);
			nvprPathCommands.resize(s);
			nvprPathCoords.resize(s);

		}

		void clear() {

			contourIndex.clear();
			contourNumber.clear();

			fillType.clear();
			fillColor.clear();
			fillIndex.clear();
			fillOpacity.clear();

			fillRule.clear();

			strokeType.clear();
			strokeColor.clear();
			strokeIndex.clear();
			strokeOpacity.clear();

			strokeWidth.clear();
			strokeLineCap.clear();
			strokeLineJoin.clear();
			strokeMiterLimit.clear();

			gradientHref.clear();

			svgString.clear();
			nvprPathCommands.clear();
			nvprPathCoords.clear();

		}
	};

	struct GradientData {
		std::vector<int> gradient_ramp_texture;
		std::vector<int> gradient_table_texture;

		std::vector<Gradient> svg_gradients;
		std::map<std::string, uint32_t> svg_gradientMap;

		void resize(size_t ramp_size, size_t table_size) {
			gradient_ramp_texture.resize(ramp_size);
			gradient_table_texture.resize(table_size);
		}

		void clear() {
			gradient_ramp_texture.clear();
			gradient_table_texture.clear();
		}
	};

public:
	VertexData vertex;
	CurveData curve;
	ContourData contour;
	PathData path;
	GradientData gradient;

	int x = 0;
	int y = 0;

	int width;
	int height;

	float vp_pos_x = 0;
	float vp_pos_y = 0;
};

//} // end of namespace OldVGContainer.

// -------- -------- -------- -------- -------- -------- -------- --------

// TODO : ref classes

namespace NewVGContainer  {

// VG elements.
typedef SimpleBezierCurve VGCurve;

class VGContour {
	int curver_begin;
	int curve_end;
};

struct VGPath {

	//
	int contour_begin;
	int contour_end;

	// fill
	uint8_t fill_type;
	u8rgba fill_color;
	uint32_t fill_gradient_id;
	float fill_opacity;
	uint8_t fill_rule;

	// stroke
	uint8_t stroke_type;
	u8rgba stroke_color;
	uint32_t stroke_gradient_id;
	float stroke_opacity;

	float stroke_width;
	uint8_t stroke_line_cap;
	uint8_t stroke_line_join;
	float stroke_miter_limit;
	// stroke_dash_arrray;
	// stroke_dash_offset;
};

class VGGradient {

};

// itetrator

class VGContainer {

public:
	//uint32_t vertexNumber() const { return (uint32_t)_vertex }
	//uint32_t curveNumber()  const { return (uint32_t)curve.vertexIndex.size(); }
	//uint32_t contourNumber()  const { return (uint32_t)contour.curveIndex.size(); }
	//uint32_t pathNumber()  const { return (uint32_t)path.contourIndex.size(); }

public:
	uint32_t addVertex(int number = 1);
	uint32_t addVertex(const glm::vec2 &v);
	uint32_t addVertex(const glm::vec2 &v0, const glm::vec2 &v1);
	uint32_t addVertex(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2);
	uint32_t addVertex(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3);

	// curve without offset
	uint32_t addCurve(VGCurveType tc);
	uint32_t addLinear(const glm::vec2 &v0, const glm::vec2 &v1);
	uint32_t addQuadratic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2);
	uint32_t addCubic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3);
	uint32_t addRational(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float w);

	// curve with offset
	uint32_t addCurve(VGCurveType tc, float offset);
	uint32_t addLinear(const glm::vec2 &v0, const glm::vec2 &v1, float offset);
	uint32_t addQuadratic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float offset);
	uint32_t addCubic(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3, float offset);
	uint32_t addRational(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2, float w, float offset);

	//
	uint32_t addContour();
	uint32_t addPath();
	uint32_t addPath(int f_rule, int f_mode, u8rgba f_color, int f_index);

public:
	void clear();

public:
	VGContainer strokeToFill();
	void cutToLength();

	void reduceDegenerate();
	void mergeAdjacentPath();

public:
	void save(const std::string &) const;
	void load(const std::string &);

	void saveLuaRvg(const std::string &) const;

private:
	void saveTxt(const std::string &) const;
	void loadTxt(const std::string &);

	void saveBinary(const std::string &) const;
	void loadBinary(const std::string &);

private:

	std::vector<VGCurve> _curve;
	std::vector<VGContour> _contour;
	std::vector<VGPath> _path;
	std::vector<VGGradient> _gradient;

};

} // end of namespace NewVGContainer.

} // end of namespace Mochimazui

#endif
