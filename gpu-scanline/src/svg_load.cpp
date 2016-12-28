
// -------- -------- -------- -------- -------- -------- -------- --------

#define _CRT_SECURE_NO_WARNINGS

#include "svg.h"

#include <stack>
#include <set>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>

#include <glm/glm.hpp>

#include <mochimazui/string.h>
#include "rapidxml.hpp"

#ifndef _DEBUG
#define SVG_LOAD_THROW(str) throw std::runtime_error(str)
#else
#define SVG_LOAD_THROW(str) { \
	char line[16]; sprintf(line, "%d", __LINE__); \
	throw std::runtime_error(string("LINE: ") + line + ", in function: " \
	+ string(__FUNCTION__) + ", " + str); \
}
#endif

// -------- -------- -------- -------- -------- -------- -------- --------
namespace Mochimazui {

	using std::set;
	using std::vector;
	using std::function;
	using std::bind;
	using std::runtime_error;
	using std::unordered_map;
	using std::make_pair;
	using std::map;

	using glm::vec2;

	typedef rapidxml::xml_node<char>* NodePtr;

	using stdext::string;

	// -------- -------- -------- -------- -------- -------- -------- --------
	// -------- -------- -------- -------- -------- -------- -------- --------

	typedef std::function<void(const string &)> AttributeReadFunction;
	//typedef unordered_map<string, AttributeReadFunction&> AttributeReadFunctionMap;
	typedef map<string, AttributeReadFunction&> AttributeReadFunctionMap;

#define ATTRIBUTE_FUNC_PAIR(name, func) std::pair<string, AttributeReadFunction&> (string(name), func)

	class AttributeReader {

	public:
		AttributeReader(
			const AttributeReader &o,
			AttributeReadFunctionMap &&rmap)
			:_map(rmap)
		{
			_map.insert(o._map.begin(), o._map.end());
		}

		AttributeReader(AttributeReadFunctionMap &&rmap)
			: _map(rmap) 
		{
		}

	public:
		void read(NodePtr pn) const {
			for (auto pa = pn->first_attribute(); pa; pa = pa->next_attribute()) {
				string name = pa->name();
				auto i =_map.find(name);
				if (i != _map.end()) {
					i->second(pa->value());
				}
			}
		}

	private:
		AttributeReadFunctionMap _map;
	};

	// -------- -------- -------- -------- -------- -------- -------- --------
	// presentation attributes
	// 
	// ref :
	// http://www.w3.org/TR/2011/REC-SVG11-20110816/attindex.html
	// or 
	// REC-SVG11-20110816.pdf
	// M.2
	// 
	// used presentation attributes :
	// -------- -------- -------- --------
	//	color
	//	cursor
	//	fill-opacity
	//	fill-rule
	//	fill
	//	opacity
	//	stop-color
	//	stop-opacity
	//	stroke-dasharray
	//	stroke-dashoffset
	//	stroke-linecap
	//	stroke-linejoin
	//	stroke-miterlimit
	//	stroke-opacity
	//	stroke-width
	//	stroke
	//	visibility
	// 
	// may be specified in :
	// -------- -------- -------- --------
	//	circle
	//	defs
	//	ellipse
	//	font
	//	g
	//	glyph
	//	glyphRef
	//	image
	//	line
	//	linearGradient
	//	missing-glyph
	//	path
	//	pattern
	//	polygon
	//	polyline
	//	radialGradient
	//	rect
	//	stop
	//	svg
	//	text
	//	textPath
	//	use
	//
	// -------- -------- -------- -------- -------- -------- -------- --------
	// 'transfrom' is a regular attribute, we also put it in this struct.
	// transform may appear in :
	// ¡®a¡¯, ¡®circle¡¯, ¡®clipPath¡¯, ¡®defs¡¯, ¡®ellipse¡¯, ¡®foreignObject¡¯, 
	// ¡®g¡¯, ¡®image¡¯, ¡®line¡¯, ¡®path¡¯, ¡®polygon¡¯, ¡®polyline¡¯, ¡®rect¡¯, 
	// ¡®switch¡¯, ¡®text¡¯, ¡®use¡¯

#define STROKE_CAP_BUTT 0
#define STROKE_CAP_ROUND 1
#define STROKE_CAP_SQUARE 2

#define STROKE_JOIN_MITER 0
#define STROKE_JOIN_ROUND 1
#define STROKE_JOIN_BEVEL 2

	struct SVGPresentationAttribute {

		float opacity;

		VGFillType fill = FT_COLOR;
		uint8_t fillRule = FR_NON_ZERO;
		u8rgba fillColor;
		uint32_t fillIndx = 0xFFFFFFFF;
		float fillOpacity = 1.f;

		VGStrokeType stroke = ST_NONE;
		u8rgba strokeColor;
		uint32_t strokeIndex = 0xFFFFFFFF;
		float strokeOpacity = 1.f;
		float strokeWidth = 1.f;

		uint32_t strokeLineCap = STROKE_CAP_BUTT;
		uint32_t strokeLineJoin = STROKE_JOIN_MITER;
		float strokeMiterLimit = 4.f;

		glm::mat3x3 transformMatrix;
	};

	// -------- -------- -------- -------- -------- -------- -------- --------
	inline int __float_as_int(float a){
		union U{ float a; int b; };
		U u;
		u.a = a;
		return u.b;
	}

	// -------- -------- -------- -------- -------- -------- -------- --------
	void SVG::load(const string &fileName, bool gen_nvpr_path_commands) {

		//if (Config::Verbose()) {
		//	printf("SVG::load: \"%s\"\n", fileName.c_str());
		//}

		FILE *fin;
		fin = fopen(fileName.c_str(), "rb");
		if (!fin) {
			SVG_LOAD_THROW("cannot open input file \"" + fileName + "\"");
		}

		fseek(fin, 0, SEEK_END);
		long size = ftell(fin);
		char *svg_text = new char[size + 1];
		if (!svg_text) {
			SVG_LOAD_THROW("SVG::load not enough memory.");
		}
		fseek(fin, 0, SEEK_SET);
		size_t size_read = fread(svg_text, 1, size, fin);
		svg_text[size] = '\0';

		// parse 
		rapidxml::xml_document<> doc;
		try {
			doc.parse<0>(svg_text);
		}
		catch (rapidxml::parse_error &e) {
			printf("SVG::load: parse error: \n%s\n", e.what());
			char *pwhere = e.where<char>();
			char *where_str = new char[256];
			strncpy(where_str, where_str, strnlen(pwhere, 255));
			printf("at: \n%s\n", where_str);
			delete[] where_str;
			SVG_LOAD_THROW("parser_error.");
		}

		// -------- -------- -------- -------- -------- -------- -------- --------
		_spVGContainer.reset(new VGContainer);
		auto &vg = *_spVGContainer;

		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// records.
		//

		std::stack<SVGPresentationAttribute> attributeStack;

		map<string, string> idToNodeType;

		string currentNodeName;
		string currentNodeId;

		int currentGradientType = GT_Linear;
		Gradient currentGradient;
		GradientStop currentStop;

		// -------- -------- -------- -------- -------- -------- -------- --------
		// helper.
		//
		auto pushAttribute = [&]() {
			auto &s = attributeStack;
			s.push(s.empty() ? SVGPresentationAttribute() : s.top());
		};

		auto popAttribute = [&]() {
			attributeStack.pop();
		};

		auto parseColor = [&](const string &_in_) -> u8rgba {
			string cstr = _in_;

			u8rgba color;
			if (!strncmp(cstr.c_str(), "rgb", 3)) {
				for (int i = 4; i < cstr.length(); ++i) {
					if (cstr[i] == ')') { cstr[i] = ' '; break; }
				}
				int r, g, b;
				sscanf(cstr.c_str() + 4, "%d,%d,%d", &r, &g, &b);
				color = u8rgba(r, g, b, 255);
			}
			else if (!strncmp(cstr.c_str(), "rgba", 4)) {
				for (int i = 5; i < cstr.length(); ++i) {
					if (cstr[i] == ')') { cstr[i] = ' '; break; }
				}
				int r, g, b; float a;
				sscanf(cstr.c_str() + 5, "%d,%d,%d,%f", &r, &g, &b, &a);
				color = u8rgba(r, g, b, (uint8_t)(a*255.f));
				//color = frgba(r / 255.f, g / 255.f, b / 255.f, a);
			}
			else if (cstr[0] == '#') {
				if (cstr.length() == 4) {
					int32_t r, g, b;
					sscanf(cstr.data() + 1, "%1x%1x%1x", &r, &g, &b);
					color = u8rgba(r * 17, g * 17, b * 17, 255);
				}
				else if (cstr.length() == 7) {
					int32_t r, g, b;
					sscanf(cstr.data() + 1, "%2x%2x%2x", &r, &g, &b);
					color = u8rgba(r, g, b, 255);
				}
				else {
					throw std::runtime_error("Unsupported color format : \"" + cstr + "\".");
				}
			}
			else {
				throw std::runtime_error("unknown color format : \"" + cstr + "\".");
			}

			return color;
		};

		auto parseTransform = [&](const string &_in_) -> glm::mat3x3 {

			// section 7.6 The 'transform' attribute

			// matrix(<a> <b> <c> <d> <e> <f>)
			// translate(<tx> [<ty>])
			// scale(<sx> [<sy>])
			// rotate(<rotate-angle> [<cx> <cy>])
			// skewX(<skew-angle>)
			// skewY(<skew-angle>)

			// -------- -------- -------- --------
			glm::mat3x3 tmat;

			// -------- -------- -------- --------
			enum Label {
				L_NULL,
				L_MATRIX,
				L_TRANSLATE,
				L_SCALE,
				L_ROTATE,
				L_SKEWX,
				L_SKEWY,
			};

			static const map<string, int> label_map = {
				{ "matrix", L_MATRIX },
				{ "translate", L_TRANSLATE },
				{ "scale", L_SCALE },
				{ "rotate", L_ROTATE },
				{ "skewX", L_SKEWX },
				{ "skewY", L_SKEWY },
			};

			int label = L_NULL;

			int i = 0;

			int vb = 0; // value begin
			int ve = 0; // value end

			// -------- -------- -------- --------
			auto tstr = _in_;
			wspToSpace(tstr);

			for (;;) {

				// find a label
				while (i < tstr.length() && !isalpha(tstr[i])) { ++i; }
				if (i == tstr.length()) { return tmat; }
				auto b = i;

				// find end of label
				while (i < tstr.length() && isalpha(tstr[i])) { ++i; }
				if (i == tstr.length()) { SVG_LOAD_THROW("transform"); }
				auto e = i;

				//
				auto im = label_map.find(tstr.substr(b, e - b));
				if (im == label_map.end()) { SVG_LOAD_THROW("transform"); }
				label = im->second;

				// find left bracket
				while (i < tstr.length() && tstr[i] != '(') { ++i; }
				if (i == tstr.length()) { SVG_LOAD_THROW("transform"); }
				vb = i + 1;

				// find right bracket
				while (i < tstr.length() && tstr[i] != ')') { ++i; }
				if (i == tstr.length()) { SVG_LOAD_THROW("transform"); }
				ve = i;
				tstr[ve] = '\0';

				++i;

				//
				glm::mat3x3 m;

				switch (label) {

				case L_MATRIX:
				{
					auto count = sscanf(tstr.c_str() + vb, "%f %f %f %f %f %f",
						&m[0][0], &m[0][1],
						&m[1][0], &m[1][1],
						&m[2][0], &m[2][1]);
					if (count != 6) { SVG_LOAD_THROW("transform:matrix()"); }
					break;
				}
				case L_TRANSLATE:
				{
					float tx, ty;
					auto count = sscanf(tstr.c_str() + vb, "%f %f", &tx, &ty);
					if (count == 1) {
						m[2][0] = m[2][1] = tx;
					}
					else if (count == 2) {
						m[2][0] = tx;
						m[2][1] = ty;
					}
					else {
						SVG_LOAD_THROW("transform:translate()");
					}
					break;
				}
				case L_SCALE:
				{
					float sx, sy;
					auto count = sscanf(tstr.c_str() + vb, "%f %f", &sx, &sy);
					if (count == 1) {
						m[0][0] = m[1][1] = sx;
					}
					else if (count == 2) {
						m[0][0] = sx;
						m[1][1] = sy;
					}
					else {
						SVG_LOAD_THROW("transform:scale()");
					}
					break;
				}
				case L_ROTATE:
				{
					assert(0);
					break;
				}
				case L_SKEWX:
				{
					assert(0);
					break;
				}
				case L_SKEWY:
				{
					assert(0);
					break;
				}
				default:
					assert(0);
					break;
				}

				tmat = tmat * m;
				tstr[ve] = ')';
			}

			return tmat;
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// default attribute reader
		// read presentation attributes
		// and push to stack top;
		// 

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReadFunction ar_gradient_spreadMethod = [&](const string &str) {
		};

		AttributeReadFunction ar_gradientTransform = [&](const string &str) {
			currentGradient.gradient_transform = parseTransform(str);
		};

		AttributeReadFunction ar_gradientUnits = [&](const string &str) {
			if (str == "userSpaceOnUse") {
				currentGradient.gradient_units = USER_SPACE_ON_USE;
			}
			else {
				throw std::runtime_error("ar_gradient_units");
			}
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReadFunction ar_lg_id = [&](const string &str) {
			currentGradient.href = str;
		};

		AttributeReadFunction ar_lg_x1 = [&](const string &str) {
			currentGradient.v1.x = str.toFloat();
		};

		AttributeReadFunction ar_lg_y1 = [&](const string &str) {
			currentGradient.v1.y = str.toFloat();
		};

		AttributeReadFunction ar_lg_x2 = [&](const string &str) {
			currentGradient.v2.x = str.toFloat();
		};

		AttributeReadFunction ar_lg_y2 = [&](const string &str) {
			currentGradient.v2.y = str.toFloat();
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReadFunction ar_rg_id = [&](const string &str) {
			currentGradient.href = str;
		};

		AttributeReadFunction ar_rg_cx = [&](const string &str) {
			currentGradient.c.x = str.toFloat();
		};

		AttributeReadFunction ar_rg_cy = [&](const string &str) {
			currentGradient.c.y = str.toFloat();
		};

		AttributeReadFunction ar_rg_r = [&](const string &str) {
			currentGradient.r = str.toFloat();
		};

		AttributeReadFunction ar_rg_fx = [&](const string &str) {
			currentGradient.f.x = str.toFloat();
			currentGradient.f_set = true;
		};

		AttributeReadFunction ar_rg_fy = [&](const string &str) {
			currentGradient.f.y = str.toFloat();
			currentGradient.f_set = true;
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReadFunction ar_stop_offset = [&](const string &str) {
			currentStop.offset = str.toFloat();
		};

		AttributeReadFunction ar_stop_color = [&](const string &str) {
			currentStop.color = parseColor(str);
		};

		AttributeReadFunction ar_stop_opacity = [&](const string &str) {
			currentStop.opacity = str.toFloat();
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReadFunction ar_id = [&](const string &str) {
			idToNodeType[str] = currentNodeName;
		};

		AttributeReadFunction ar_color = [&](const string &str) {
			attributeStack.pop();
		};

		AttributeReadFunction ar_fill_rule = [&](const string &str) {
			if (str == "nonzero") {
				attributeStack.top().fillRule = FR_NON_ZERO;
			}
			else if (str == "evenodd") {
				attributeStack.top().fillRule = FR_EVEN_ODD;
			}
			else {
				// ERROR.
			}
		};

		AttributeReadFunction ar_fill_opacity = [&](const string &str) {
			attributeStack.top().fillOpacity = str.toFloat();
		};

		AttributeReadFunction ar_fill = [&](const string &str) {
			if (str == "none") {
				attributeStack.top().fill = FT_NONE;
			}
			else if (str.substr(0, 3) == "url") {
				auto id = str.substr(4);
				id = id.substr(0, id.length() - 1);
				if (id[0] == '#') { id = id.substr(1); }

				auto ig = vg.gradient.svg_gradientMap.find(id);
				if (ig != vg.gradient.svg_gradientMap.end()) {
					attributeStack.top().fill = FT_GRADIENT;
					attributeStack.top().fillIndx = ig->second;
				}
				else {
					// gradient not found.
					throw std::runtime_error("SVG::load: attribute fill: url not found.");
				}
			}
			else {
				attributeStack.top().fillColor = parseColor(str);
				attributeStack.top().fill = FT_COLOR;
			}
		};

		AttributeReadFunction ar_opacity = [&](const string &str) {
			attributeStack.top().opacity = str.toFloat();
		};

		AttributeReadFunction ar_stroke_opacity = [&](const string &str) {
			attributeStack.top().strokeOpacity = str.toFloat();
		};
		AttributeReadFunction ar_stroke_width = [&](const string &str) {
			attributeStack.top().strokeWidth = str.toFloat();
		};
		AttributeReadFunction ar_stroke_linecap = [&](const string &str) {
			if (str == "butt") {
				attributeStack.top().strokeLineCap = STROKE_CAP_BUTT;
			}
			else if (str == "round") {
				attributeStack.top().strokeLineCap = STROKE_CAP_ROUND;
			}
			else if (str == "square") {
				attributeStack.top().strokeLineCap = STROKE_CAP_SQUARE;
			}
			else {
				throw std::runtime_error("SVG::load: illegal stroke-linecap \"" + str + "\"");
			}
		};
		AttributeReadFunction ar_stroke_linejoin = [&](const string &str) {
			if (str == "miter") {
				attributeStack.top().strokeLineJoin = STROKE_JOIN_MITER;
			}
			else if (str == "round") {
				attributeStack.top().strokeLineJoin = STROKE_JOIN_ROUND;
			}
			else if (str == "bevel") {
				attributeStack.top().strokeLineJoin = STROKE_JOIN_BEVEL;
			}
			else {
				throw std::runtime_error("SVG::load: illegal stroke-linejoin \"" + str + "\"");
			}
		};
		AttributeReadFunction ar_stroke_miterlimit = [&](const string &str) {
			attributeStack.top().strokeMiterLimit = str.toFloat();
		};
		AttributeReadFunction ar_stroke_dasharray = [&](const string &str) {
		};
		AttributeReadFunction ar_stroke_dashoffset = [&](const string &str) {
		};

		AttributeReadFunction ar_stroke = [&](const string &str) {
			if (str == "none") {
				attributeStack.top().stroke = ST_NONE;
			}
			else if (str.substr(0, 3) == "url") {
				auto id = str.substr(4);
				id = id.substr(0, id.length() - 1);
				if (id[0] == '#') { id = id.substr(1); }

				auto ig = vg.gradient.svg_gradientMap.find(id);
				if (ig != vg.gradient.svg_gradientMap.end()) {
					attributeStack.top().stroke = ST_GRADIENT;
					attributeStack.top().strokeIndex = ig->second;
				}
				else {
					// gradient not found.
					throw std::runtime_error("SVG::load: attribute fill: url not found.");
				}
			}
			else {
				attributeStack.top().strokeColor = parseColor(str);
				attributeStack.top().stroke = ST_COLOR;
			}
		};

		AttributeReadFunction ar_visibility = [&](const string &str) {
		};

		AttributeReadFunction ar_transform = [&](const string &str) {
			auto &mat = attributeStack.top().transformMatrix;
			mat = mat * parseTransform(str);
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReadFunction ar_path_d = [&](const string &pstr) {

			// -------- -------- -------- --------
			if (!pstr.length()) { return; }

			// -------- -------- -------- --------
			// local variables

			enum State {
				PS_CMD,
				PS_NUMBER,
			};

			// 
			int state = PS_CMD; // current state
			char cc = 0; // current char

			char prevCmd = 0;
			char cmd = 0;

			char nstr[256]; // number str
			int nstr_len = 0;

			float value[7];
			int value_index = 0;

			bool relative = false;

			//glm::vec2 cp;
			glm::vec2 lp[2];
			//int last_point = 0;

			// -------- -------- -------- --------
			// helper functions

			auto getPoint = [&](float *a, vec2 &p) {
				p.x = a[0];
				p.y = a[1];
				if (relative) {
					p += lp[0];
				}
			};

			auto getX = [&]() -> float {
				return relative ? lp[0].x + value[0] : value[0];
			};

			auto getY = [&]() -> float {
				return relative ? lp[0].y + value[0] : value[0];
			};

			// -------- -------- -------- --------
			// start
			auto pid = vg.addPath();
			auto contour_id = vg.addContour();
			auto contour_first_vertex_id = vg.vertexNumber();

			vg.path.svgString[pid] = pstr;

			if (gen_nvpr_path_commands) {
				vg.path.nvprPathCommands.push_back(vector<uint8_t>());
				vg.path.nvprPathCoords.push_back(vector<float>());
			}

			auto push_nvpr_coord = [&](const glm::vec2 &v) {
				if (gen_nvpr_path_commands) {
					vg.path.nvprPathCoords.back().push_back(v.x);
					vg.path.nvprPathCoords.back().push_back(v.y);
				}
			};

			int i = 0;
			for (;;) {
				if (i > pstr.length()) { break; }
				else if (i == pstr.length()) {
					cc = '\0';
				}
				else {
					cc = pstr[i];
				}

				//
				switch (state) {

				case PS_CMD:
				{
					// ----
					relative = !!islower(cc);
					char lcc = tolower(cc);
					switch (lcc) {
					case 'm':
						// new contour 
						//if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back('M'); }
						if (vg.contour.curveNumber[contour_id]) {
							contour_id = vg.addContour();
							contour_first_vertex_id = vg.vertexNumber();
						}
						cmd = lcc;
						state = PS_NUMBER;						
						break;
					case 'l':
					case 'h':
					case 'v':
					case 'q':
					case 'c':
					case 's':
					case 'a':
						//if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper(lcc)); }
						cmd = lcc;
						state = PS_NUMBER;						
						break;
					case '\0':
					case ' ':
					case '\n':
					case '\r':
						break;
					case 'z':
					{
						// close contour.
						//if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('Z')); }
						if (contour_first_vertex_id == vg.vertexNumber()) { break; }
						auto fv = vg.vertex.point[contour_first_vertex_id];
						auto lv = vg.vertex.point.back();
						if (fv != lv) { 
							vg.addLinear(lv, fv); 
						}
						vg.contour.closed[contour_id] = 1;
						break;
					}
					default:
						state = PS_CMD;
						break;
					}
					++i;
					break;
				}
				case PS_NUMBER:
				{
					if (
						(nstr_len != 0 && tolower(nstr[nstr_len-1]) == 'e' && cc == '-' ) ||
						(isdigit(cc) || cc == 'e' || cc == 'E' || cc == '.') ||
						(nstr_len == 0 && cc == '-')
						) {
						nstr[nstr_len] = cc;
						++nstr_len;
						++i;
					}
					else {
						if (nstr_len) {
							// end this number
							nstr[nstr_len] = '\0';
							sscanf(nstr, "%f", value + value_index);
							++value_index;
							nstr_len = 0;
							// 
							switch (cmd) {
								// - move
							case 'm':
								if (value_index == 2) {
									getPoint(value, lp[0]);
									value_index = 0;

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('M')); }
									push_nvpr_coord(lp[0]);
								}
								break;
								// - line
							case 'l':
								if (value_index == 2) {
									vg.addCurve(CT_Linear);
									auto vi = vg.addVertex(2);
									vg.vertex.point[vi] = lp[0];
									getPoint(value, vg.vertex.point[vi + 1]);
									lp[0] = vg.vertex.point[vi + 1];
									value_index = 0;
									vg.reduceDegenerate();

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('L')); }
									push_nvpr_coord(vg.vertex.point[vi + 1]);
								}
								break;
							case 'h':
								if (value_index == 1) {
									vg.addCurve(CT_Linear);
									auto vi = vg.addVertex(2);
									vg.vertex.point[vi] = lp[0];
									vg.vertex.point[vi + 1] = vec2(getX(), lp[0].y);
									lp[0] = vg.vertex.point[vi + 1];
									value_index = 0;
									vg.reduceDegenerate();

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('L')); }
									push_nvpr_coord(vg.vertex.point[vi + 1]);
								}
								break;
							case 'v':
								if (value_index == 1) {
									vg.addCurve(CT_Linear);
									auto vi = vg.addVertex(2);
									vg.vertex.point[vi] = lp[0];
									vg.vertex.point[vi + 1] = vec2(lp[0].x, getY());
									lp[0] = vg.vertex.point[vi + 1];
									value_index = 0;
									vg.reduceDegenerate();

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('L')); }
									push_nvpr_coord(vg.vertex.point[vi + 1]);
								}
								break;
								// - cubic curve
							case 'c':
								if (value_index == 6) {
									vg.addCurve(CT_Cubic);
									auto vi = vg.addVertex(4);
									vg.vertex.point[vi] = lp[0];
									getPoint(value, vg.vertex.point[vi + 1]);
									getPoint(value + 2, vg.vertex.point[vi + 2]);
									getPoint(value + 4, vg.vertex.point[vi + 3]);
									lp[0] = vg.vertex.point[vi + 3];
									lp[1] = vg.vertex.point[vi + 2];
									value_index = 0;
									vg.reduceDegenerate();

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('C')); }
									push_nvpr_coord(vg.vertex.point[vi + 1]);
									push_nvpr_coord(vg.vertex.point[vi + 2]);
									push_nvpr_coord(vg.vertex.point[vi + 3]);
								}
								break;
							case 's':
								if (value_index == 4) {
									vg.addCurve(CT_Cubic);
									auto vi = vg.addVertex(4);
									vg.vertex.point[vi] = lp[0];
									vg.vertex.point[vi + 1] = lp[0] + (lp[0] - lp[1]);
									getPoint(value, vg.vertex.point[vi + 2]);
									getPoint(value + 2, vg.vertex.point[vi + 3]);
									lp[0] = vg.vertex.point[vi + 3];
									lp[1] = vg.vertex.point[vi + 2];
									value_index = 0;
									vg.reduceDegenerate();

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('C')); }
									push_nvpr_coord(vg.vertex.point[vi + 1]);
									push_nvpr_coord(vg.vertex.point[vi + 2]);
									push_nvpr_coord(vg.vertex.point[vi + 3]);
								}
								break;
								// - quad curve
							case 'q':
								if (value_index == 4) {
									vg.addCurve(CT_Quadratic);
									auto vi = vg.addVertex(3);
									vg.vertex.point[vi] = lp[0];
									getPoint(value, vg.vertex.point[vi + 1]);
									getPoint(value + 2, vg.vertex.point[vi + 2]);
									lp[0] = vg.vertex.point[vi + 2];
									lp[1] = vg.vertex.point[vi + 1];
									value_index = 0;
									vg.reduceDegenerate();

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('Q')); }
									push_nvpr_coord(vg.vertex.point[vi + 1]);
									push_nvpr_coord(vg.vertex.point[vi + 2]);
								}
								break;
							case 't':
								if (value_index == 2) {
									vg.addCurve(CT_Quadratic);
									auto vi = vg.addVertex(3);
									vg.vertex.point[vi] = lp[0];
									vg.vertex.point[vi + 1] = lp[0] + (lp[0] - lp[1]);
									getPoint(value, vg.vertex.point[vi + 2]);
									lp[0] = vg.vertex.point[vi + 2];
									lp[1] = vg.vertex.point[vi + 1];
									value_index = 0;
									vg.reduceDegenerate();

									if (gen_nvpr_path_commands) { vg.path.nvprPathCommands.back().push_back(toupper('Q')); }
									push_nvpr_coord(vg.vertex.point[vi + 1]);
									push_nvpr_coord(vg.vertex.point[vi + 2]);
								}
								break;
								// - arc
							case 'a':
								if (value_index == 7) {

									// read arc.
									// (rx ry x-axis-rotation large-arc-flag sweep-flag x y)+
									glm::vec2 v_start = lp[0];

									float rx = value[0];
									float ry = value[1];

									float x_axis_rotation = value[2];
									int large_arc_flag = (int)value[3];
									int sweep_flag = (int)value[4];

									glm::vec2 v_end;
									getPoint(value + 5, v_end);

									vg.addSVGArc(v_start, rx, ry, x_axis_rotation,
										large_arc_flag, sweep_flag, v_end);

									if (gen_nvpr_path_commands) {
										vg.path.nvprPathCoords.back().push_back(rx);
										vg.path.nvprPathCoords.back().push_back(ry);
										vg.path.nvprPathCoords.back().push_back((float)x_axis_rotation);
										vg.path.nvprPathCoords.back().push_back((float)large_arc_flag);
										vg.path.nvprPathCoords.back().push_back((float)sweep_flag);
										vg.path.nvprPathCoords.back().push_back(v_end.x);
										vg.path.nvprPathCoords.back().push_back(v_end.y);
									}

									lp[0] = *vg.vertex.point.rbegin();
									lp[1] = *(vg.vertex.point.rbegin() + 1);

									value_index = 0;
									vg.reduceDegenerate();
								}
								break;
								// - default
							default:
								assert(0);
								break;
							}
						} // END OF if (nstr_len) {}

						// next state?
						switch (tolower(cc)) {
						case ' ':
						case '\t':
						case '\n':
						case '\r':
						case ',':
						case '\0':
							// skip char
							++i;
							break;
						case 'm':
						case 'l':
						case 'h':
						case 'v':
						case 'q':
						case 'c':
						case 's':
						case 'a':
						case 'z':
							// reserve this char and goto CMD.
							if (gen_nvpr_path_commands && (tolower(cc)=='z')) { 
								vg.path.nvprPathCommands.back().push_back(toupper('Z')); 
							}
							state = PS_CMD;
							break;
						case '-':
							// reserve this char and stay NUMBER;
							break;
						default:
							assert(0);
							break;
						}
					}
				}
				default:
				{
					break;
				}
				}
			}
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReadFunction ar_svg_width = [&](const string &str) {
			float w;
			auto count = sscanf(str.c_str(), "%f", &w);
			if (!count) { SVG_LOAD_THROW("width"); }
			vg.width = _width = (uint32_t)w;
		};

		AttributeReadFunction ar_svg_height = [&](const string &str) {
			float h;
			auto count = sscanf(str.c_str(), "%f", &h);
			if (!count) { SVG_LOAD_THROW("height"); }
			vg.height = _height = (uint32_t)h;
		};

		AttributeReadFunction ar_svg_viewbox = [&](const string &str) {
			auto  count = sscanf(str.c_str(), "%f %f %f %f",
				&_viewBox[0].x, &_viewBox[0].y,
				&_viewBox[1].x, &_viewBox[1].y);
			if (count != 4) { SVG_LOAD_THROW("viewBox"); }
		};

		AttributeReadFunction ar_style = [&](const string &style) {
			auto style_list = style.split(';');
			for (auto s : style_list) {
				auto m = s.split(':');
				auto key = m[0];
				auto value = m[1];

				if (key == "stop-color") {
					currentStop.color = parseColor(value);
				}
				else if (key == "stop-opacity") {
					currentStop.opacity = value.toFloat();
				}
				else {
					throw std::runtime_error(key);
				}
			}
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReader defaultAttributeReader(
			AttributeReadFunctionMap{
			ATTRIBUTE_FUNC_PAIR("color", ar_color),

			ATTRIBUTE_FUNC_PAIR("fill-rule", ar_fill_rule),
			ATTRIBUTE_FUNC_PAIR("fill-opacity", ar_fill_opacity),
			ATTRIBUTE_FUNC_PAIR("fill", ar_fill),

			ATTRIBUTE_FUNC_PAIR("opacity", ar_opacity),

			ATTRIBUTE_FUNC_PAIR("stroke-dasharray", ar_stroke_dasharray),
			ATTRIBUTE_FUNC_PAIR("stroke-dashoffset", ar_stroke_dashoffset),
			ATTRIBUTE_FUNC_PAIR("stroke-linecap", ar_stroke_linecap),
			ATTRIBUTE_FUNC_PAIR("stroke-linejoin", ar_stroke_linejoin),
			ATTRIBUTE_FUNC_PAIR("stroke-miterlimit", ar_stroke_miterlimit),
			ATTRIBUTE_FUNC_PAIR("stroke-opacity", ar_stroke_opacity),
			ATTRIBUTE_FUNC_PAIR("stroke-width", ar_stroke_width),
			ATTRIBUTE_FUNC_PAIR("stroke", ar_stroke),

			ATTRIBUTE_FUNC_PAIR("visibility", ar_visibility),
			ATTRIBUTE_FUNC_PAIR("transform", ar_transform),

			ATTRIBUTE_FUNC_PAIR("style", ar_style),
		}
		);

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReader pathAttributeReader(
			defaultAttributeReader,
			AttributeReadFunctionMap{
				ATTRIBUTE_FUNC_PAIR("d", ar_path_d)
		}
		);

		// -------- -------- -------- -------- -------- -------- -------- --------
		AttributeReader svgAttributeReader(
			//defaultAttributeReader,
			AttributeReadFunctionMap{
				ATTRIBUTE_FUNC_PAIR("width", ar_svg_width),
				ATTRIBUTE_FUNC_PAIR("height", ar_svg_height),
				ATTRIBUTE_FUNC_PAIR("viewBox", ar_svg_viewbox),
		}
		);

		AttributeReader linearGradientAttributeReader(
			defaultAttributeReader,
			AttributeReadFunctionMap{
				ATTRIBUTE_FUNC_PAIR("id", ar_lg_id),
				ATTRIBUTE_FUNC_PAIR("x1", ar_lg_x1),
				ATTRIBUTE_FUNC_PAIR("y1", ar_lg_y1),
				ATTRIBUTE_FUNC_PAIR("x2", ar_lg_x2),
				ATTRIBUTE_FUNC_PAIR("y2", ar_lg_y2),
				ATTRIBUTE_FUNC_PAIR("gradientTransform", ar_gradientTransform),
				ATTRIBUTE_FUNC_PAIR("gradientUnits", ar_gradientUnits),
		}
		);

		AttributeReader radialGradientAttributeReader(
			defaultAttributeReader,
			AttributeReadFunctionMap{
				ATTRIBUTE_FUNC_PAIR("id", ar_rg_id),
				ATTRIBUTE_FUNC_PAIR("cx", ar_rg_cx),
				ATTRIBUTE_FUNC_PAIR("cy", ar_rg_cy),
				ATTRIBUTE_FUNC_PAIR("r", ar_rg_r),
				ATTRIBUTE_FUNC_PAIR("fx", ar_rg_fx),
				ATTRIBUTE_FUNC_PAIR("fy", ar_rg_fy),
				ATTRIBUTE_FUNC_PAIR("gradientTransform", ar_gradientTransform),
				ATTRIBUTE_FUNC_PAIR("gradientUnits", ar_gradientUnits),
		}
		);

		AttributeReader stopAttributeReader(
			defaultAttributeReader,
			AttributeReadFunctionMap{
				ATTRIBUTE_FUNC_PAIR("offset", ar_stop_offset),
				ATTRIBUTE_FUNC_PAIR("stop-color", ar_stop_color),
				ATTRIBUTE_FUNC_PAIR("stop-opacity", ar_stop_opacity),
		}
		);

		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		// -------- -------- -------- -------- -------- -------- -------- --------
		//

		typedef std::function<void(NodePtr)> NodeReadFunction;
		//std::unordered_map<string, NodeReadFunction& > nodeReaders;
		std::map<string, NodeReadFunction& > nodeReaders;

		// -------- -------- -------- -------- -------- -------- -------- --------
		NodeReadFunction readSVG;

		NodeReadFunction readDefault;
		NodeReadFunction readNothing = [](NodePtr){};

		NodeReadFunction readLinearGradient;
		NodeReadFunction readRadialGradient;
		NodeReadFunction readStop;

		NodeReadFunction readGroup;

		NodeReadFunction readPath;
		NodeReadFunction readLine;
		NodeReadFunction readRect;
		NodeReadFunction readPolygon;
		NodeReadFunction readCircle;
		NodeReadFunction readEllipse;

		NodeReadFunction readStyle;

		// -------- -------- -------- -------- -------- -------- -------- --------
		nodeReaders = decltype(nodeReaders)({
			{ "<xmlattr>", readNothing },
			{ "<xmlcomment>", readNothing },

			{ "defs", readDefault },
			{ "g", readGroup },
			{ "path", readPath },
			{ "rect", readLine },
			{ "rect", readRect },
			{ "polygon", readPolygon },
			{ "circle", readRect },
			{ "ellipse", readRect },

			{ "linearGradient", readLinearGradient },
			{ "radialGradient", readRadialGradient },
			{ "stop", readStop },
		});

		// -------- -------- -------- -------- -------- -------- -------- --------
		readLinearGradient = [&](NodePtr pn) {
			pushAttribute();
			currentGradient.clear();

			currentGradient.gradient_type = GT_Linear;
			linearGradientAttributeReader.read(pn);
			readDefault(pn);

			vg.gradient.svg_gradients.push_back(currentGradient);
			vg.gradient.svg_gradientMap[currentGradient.href] = (uint32_t)vg.gradient.svg_gradients.size() - 1;
			popAttribute();
		};

		readRadialGradient = [&](NodePtr pn) {
			pushAttribute();			
			currentGradient.clear();

			currentGradient.gradient_type = GT_Radial;
			radialGradientAttributeReader.read(pn);
			readDefault(pn);

			if (!currentGradient.f_set) {
				currentGradient.f = currentGradient.c;
			}

			vg.gradient.svg_gradients.push_back(currentGradient);
			vg.gradient.svg_gradientMap[currentGradient.href] = (uint32_t)vg.gradient.svg_gradients.size() - 1;
			popAttribute();
		};

		readStop = [&](NodePtr pn) {
			pushAttribute();
			stopAttributeReader.read(pn);

//#define USE_PRE_MULTIPLIED_ALPHA
#ifdef USE_PRE_MULTIPLIED_ALPHA
			currentStop.color.r *= currentStop.color.a / 255.f * currentStop.opacity;
			currentStop.color.g *= currentStop.color.a / 255.f * currentStop.opacity;
			currentStop.color.b *= currentStop.color.a / 255.f * currentStop.opacity;
			currentStop.color.a *= currentStop.opacity;
#else
			currentStop.color.a = (uint8_t)(currentStop.color.a * currentStop.opacity);
#endif
			currentGradient.gradient_stops.push_back(currentStop);
			popAttribute();
		};

		readPath = [&](NodePtr pn) {

			auto vbegin = vg.vertexNumber();

			pushAttribute();
			pathAttributeReader.read(pn);

			auto &a = attributeStack.top();
			auto pi = vg.pathNumber() - 1;
			vg.path.fillType[pi] = a.fill;
			vg.path.fillRule[pi] = a.fillRule;

#ifdef USE_PRE_MULTIPLIED_ALPHA
			a.fillColor.r *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.g *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.b *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.a *= a.fillOpacity;
#else
			a.fillColor.a = (uint8_t)(a.fillColor.a * a.fillOpacity);

			if (_a128) {
				a.fillColor.a = (uint8_t)((a.fillColor.a / 255.f) * 127.f);
				a.fillColor.a = (uint8_t)((a.fillColor.a / 127.f) * 255.f);
			}
#endif

			vg.path.fillColor[pi] = a.fillColor;
			vg.path.fillIndex[pi] = a.fillIndx;
			vg.path.fillOpacity[pi] = a.fillOpacity;

			vg.path.strokeType[pi] = a.stroke;
			vg.path.strokeColor[pi] = a.strokeColor;
			vg.path.strokeIndex[pi] = a.strokeIndex;
			vg.path.strokeOpacity[pi] = a.strokeOpacity;

			vg.path.strokeWidth[pi] = a.strokeWidth;
			vg.path.strokeLineCap[pi] = a.strokeLineCap;
			vg.path.strokeLineJoin[pi] = a.strokeLineJoin;
			vg.path.strokeMiterLimit[pi] = a.strokeMiterLimit;
			// dash array
			// dash offset

			// transform all points in the last path.
			auto vend = vg.vertexNumber();
			auto &mat = attributeStack.top().transformMatrix;

			for (int i = vbegin; i < (int)vend; ++i) {
				auto &v = vg.vertex.point[i];
				auto v3 = mat * glm::vec3(v.x, v.y, 1);
				v3 /= v3.z;
				v = glm::vec2(v3.x, v3.y);
			}

			popAttribute();
		};

		readPolygon = [&](NodePtr pn) {
			auto vbegin = vg.vertexNumber();

			pushAttribute();
			pathAttributeReader.read(pn);

			vg.addPath();
			vg.addContour();

			// read polygon attribute
			for (auto pa = pn->first_attribute(); pa; pa = pa->next_attribute()) {
				if (pa->name() == string("points")) {

					string value = pa->value();
					for (auto &c : value) {
						if (c == ',') { c = ' '; }
					}
					std::istringstream iss(value);

					glm::vec2 vf, v0, v1;
					iss >> vf.x >> vf.y;
					v0 = vf;

					while (iss >> v1.x >> v1.y) {
						vg.addLinear(v0, v1);
						v0 = v1;
					}
					vg.addLinear(v0, vf);
				}
			}

			auto &a = attributeStack.top();
			auto pi = vg.pathNumber() - 1;
			vg.path.fillType[pi] = a.fill;
			vg.path.fillRule[pi] = a.fillRule;

#ifdef USE_PRE_MULTIPLIED_ALPHA
			a.fillColor.r *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.g *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.b *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.a *= a.fillOpacity;
#else
			a.fillColor.a = (uint8_t)(a.fillColor.a * a.fillOpacity);

			if (_a128) {
				a.fillColor.a = (uint8_t)((a.fillColor.a / 255.f) * 127.f);
				a.fillColor.a = (uint8_t)((a.fillColor.a / 127.f) * 255.f);
			}
#endif

			vg.path.fillColor[pi] = a.fillColor;
			vg.path.fillIndex[pi] = a.fillIndx;
			vg.path.fillOpacity[pi] = a.fillOpacity;

			vg.path.strokeType[pi] = a.stroke;
			vg.path.strokeColor[pi] = a.strokeColor;
			vg.path.strokeIndex[pi] = a.strokeIndex;
			vg.path.strokeWidth[pi] = a.strokeWidth;

			// transform all points in the last path.
			auto vend = vg.vertexNumber();
			auto &mat = attributeStack.top().transformMatrix;

			for (int i = vbegin; i < (int)vend; ++i) {
				auto &v = vg.vertex.point[i];
				auto v3 = mat * glm::vec3(v.x, v.y, 1);
				v3 /= v3.z;
				v = glm::vec2(v3.x, v3.y);
			}

			popAttribute();
		};

		readLine = [&](NodePtr pn) {
			pushAttribute();
			// TODO: read
			throw std::runtime_error("SVG::load read line");
			popAttribute();
		};

		readRect = [&](NodePtr pn) {

			auto vbegin = vg.vertexNumber();

			pushAttribute();
			pathAttributeReader.read(pn);

			// read rect attribute
			float x, y, w, h;
			for (auto pa = pn->first_attribute(); pa; pa = pa->next_attribute()) {
				if (pa->name() == string("x")) {
					x = string(pa->value()).toFloat();
				} if (pa->name() == string("y")) {
					y = string(pa->value()).toFloat();
				} if (pa->name() == string("width")) {
					w = string(pa->value()).toFloat();
				} if (pa->name() == string("height")) {
					h = string(pa->value()).toFloat();
				}
			}

			vg.addPath();
			vg.addContour();

			glm::vec2 v[4] = {
				glm::vec2(x, y),
				glm::vec2(x, y + h),
				glm::vec2(x + w, y + h),
				glm::vec2(x + w, y),
			};

			vg.addLinear(v[0], v[1]);
			vg.addLinear(v[1], v[2]);
			vg.addLinear(v[2], v[3]);
			vg.addLinear(v[3], v[0]);

			auto &a = attributeStack.top();
			auto pi = vg.pathNumber() - 1;
			vg.path.fillType[pi] = a.fill;
			vg.path.fillRule[pi] = a.fillRule;

#ifdef USE_PRE_MULTIPLIED_ALPHA
			a.fillColor.r *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.g *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.b *= a.fillColor.a / 255.f* a.fillOpacity;
			a.fillColor.a *= a.fillOpacity;
#else
			a.fillColor.a = (uint8_t)(a.fillColor.a * a.fillOpacity);

			if (_a128) {
				a.fillColor.a = (uint8_t)((a.fillColor.a / 255.f) * 127.f);
				a.fillColor.a = (uint8_t)((a.fillColor.a / 127.f) * 255.f);
			}
#endif

			vg.path.fillColor[pi] = a.fillColor;
			vg.path.fillIndex[pi] = a.fillIndx;
			vg.path.fillOpacity[pi] = a.fillOpacity;

			vg.path.strokeType[pi] = a.stroke;
			vg.path.strokeColor[pi] = a.strokeColor;
			vg.path.strokeIndex[pi] = a.strokeIndex;
			vg.path.strokeWidth[pi] = a.strokeWidth;

			// transform all points in the last path.
			auto vend = vg.vertexNumber();
			auto &mat = attributeStack.top().transformMatrix;

			for (int i = vbegin; i < (int)vend; ++i) {
				auto &v = vg.vertex.point[i];
				auto v3 = mat * glm::vec3(v.x, v.y, 1);
				v3 /= v3.z;
				v = glm::vec2(v3.x, v3.y);
			}

			popAttribute();
		};

		readCircle = [&](NodePtr pn) {
			pushAttribute();
			// TODO: read
			throw std::runtime_error("SVG::load read circle");
			popAttribute();
		};

		readEllipse = [&](NodePtr pn) {
			pushAttribute();
			// TODO: read
			throw std::runtime_error("SVG::load read ellipse");
			popAttribute();
		};

		readGroup = [&](NodePtr pn) {
			pushAttribute();
			defaultAttributeReader.read(pn);
			readDefault(pn);
			popAttribute();
		};

		readDefault = [&](NodePtr pn) {
			// psn -> pointer to sub node
			for (auto psn = pn->first_node(); psn; psn = psn->next_sibling()) {
				string name = psn->name();
				currentNodeName = name;
				auto i = nodeReaders.find(name);
				if (i != nodeReaders.end()) {
					i->second(psn);
				}
				else {
					// 
				}
			}
		};

		readSVG = [&](NodePtr pn) {
			pushAttribute();
			svgAttributeReader.read(pn);
			readDefault(pn);
			popAttribute();
		};

		readStyle = [&](NodePtr pn) {
		};

		// -------- -------- -------- -------- -------- -------- -------- --------
		try{
			for (auto node = doc.first_node(); node; node = node->next_sibling()) {
				if (node->name_size() == 3 && !strcmp(node->name(), "svg")) {
					readSVG(node);
				}
			}
		}
		catch (std::exception &e) {
			printf("exception in SVG::load():%d: %s\n", __LINE__, e.what());
		}

		// -------- -------- -------- -------- -------- -------- -------- --------
		delete[] svg_text;

		// -------- -------- -------- -------- -------- -------- -------- --------
		vg.width = _width = (int)_viewBox[1].x;
		vg.height = _height = (int)_viewBox[1].y;

		vg.vp_pos_x = _viewBox[0].x;
		vg.vp_pos_y = _viewBox[0].y;

		for (auto &v : vg.vertex.point) {
			//v -= vec2(_viewBox[0].x, _viewBox[1].x);
			v.x -= _viewBox[0].x;
			v.y -= _viewBox[0].y;

			v.y = _height - v.y;
		}

		_viewBox[0] = vec2(0, 0);
	}

}
