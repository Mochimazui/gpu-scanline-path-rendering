 
#define _CRT_SECURE_NO_WARNINGS

#include "rvg.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <memory>

#include <glm/vec2.hpp>

#include <cstdio>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

#include "vg_config.h"
#include "gradient.h"

namespace Mochimazui {
	
	inline uint32_t f32tou8(float a){
		int ret=(int)(a*255.f);
		if(ret>255){ret=255;}
		if(ret<0){ret=0;}
		return ret;
	}

	inline int __float_as_int(float a){
		union U{float a;int b;};
		U u;
		u.a=a;
		return u.b;
	}

	inline float safeRcp(float a){
		return a?1.f/a:0.f;
	}

	inline float lerp(float a,float b,float t){
		return a+(b-a)*t;
	}

	using std::cout;
	using std::endl;

	using std::ifstream;
	using std::runtime_error;
	using std::getline;

	using std::shared_ptr;

	using glm::vec2;

	using Mochimazui::u8rgba;
	using Mochimazui::frgba;

	void RVG::load(const string &fileName) {

		//if (Config::Verbose()) {
		//	printf("RVG::load: \"%s\"\n", fileName.c_str());
		//}

		ifstream fin;
		fin.open(fileName);
		if (!fin) {
			throw runtime_error("RVG::load(): can not open file \"" + fileName + "\"");
		}

		// --------- --------- --------- --------- --------- --------- --------- ---------
		_spVGContainer.reset(new VGContainer);
		auto &vg = *_spVGContainer;

		// --------- --------- --------- --------- --------- --------- --------- ---------
		// count line number;

		auto lineNumber = [&fileName]()-> uint32_t{
			//if (Config::Verbose()) {
			//	cout << "RVG::load() count line number." << endl;
			//}

			ifstream fin;
			fin.open(fileName);

			auto begin = fin.tellg();
			fin.seekg(0, std::ios::end);
			auto end = fin.tellg();
			auto len = end - begin;
			char *c = new char[len];
			fin.seekg(0, std::ios::beg);
			fin.read(c, len);

			uint32_t ln = 0;
			for (int i = 0; i < len; ++i) {
				if (c[i] == '\n') { ++ln; }
			}
			delete[]c;
			ln -= 3;
			//if (Config::Verbose()) {
			//	cout << "RVG::load() line number is " << ln << endl;
			//}
			return ln;
		}();

		[&fileName, this](){
			ifstream fin;
			fin.open(fileName);
			string tstr;
			std::getline(fin, tstr); _header = tstr + "\n";
			std::getline(fin, tstr); _header += tstr + "\n";
			std::getline(fin, tstr); _header += tstr + "\n";

			_lines.clear();
			while (std::getline(fin, tstr)) {
				if (tstr.find("//") == string::npos) {
					_lines.push_back(tstr);
				}
			}
		}();
		
		int currentLine = 0;

		// --------- --------- --------- --------- --------- --------- --------- ---------
		// 

		string tstr;
		tstr.reserve(65536);

		auto replace_comma = [](string &str) {
			for (auto &c : str) { if (c == ',') { c = ' '; } }
		};

		auto replace_colon = [](string &str) {
			for (auto &c : str) { if (c == ':') { c = ' '; } }
		};

		auto read_point = [&]() -> vec2 {
			string istr;
			fin >> istr;
			replace_comma(istr);
			replace_colon(istr);

			vec2 v;
			sscanf(istr.c_str(), "%f %f", &v.x, &v.y);

			//std::istringstream iss(istr);
			//iss >> v.x >> v.y;

			return v;
		};
		
		// --------- --------- --------- --------- --------- --------- --------- ---------
		
		fin >> tstr;
		_viewport[0] = read_point();
		_viewport[1] = read_point();

		fin >> tstr;
		_window[0] = read_point();
		_window[1] = read_point();

		//
		fin >> tstr;
		assert(tstr == "scene");
		fin >> tstr;
		assert(tstr == "dyn_identity");

		std::vector<float> temp_knots;
		std::vector<int> temp_ramp_buffer;
		std::vector<TGradientItem> temp_grad_items;

		int last_path_vertex_number = -1;

		//
		while (fin >> tstr) {

			++currentLine;

			if (tstr == "//") {
				// ignore;
				getline(fin, tstr);
				continue;
			}

			if (tstr != "1") {
				assert(0);
			}

			vec2 refPoint;
			vec2 p[4];
			char lastOpChar = '\0';

			// 
			fin >> tstr;
			assert(tstr == "element");

			int path, contour;
			if (last_path_vertex_number != vg.vertex.point.size()) {
				path = vg.addPath();
				contour = vg.addContour();
				last_path_vertex_number = (int)vg.vertex.point.size();
			}

			// 
			string fillRule;
			fin >> fillRule;
			if (fillRule == "nzfill") {
				vg.path.fillRule[path] = 0;
			}
			else if(fillRule == "ofill") {
				vg.path.fillRule[path] = 1;
			}
			else {
				getline(fin, tstr);
				continue;
			}

			// 
			fin >> tstr;
			if (tstr == "depth") { getline(fin, tstr); continue; }
			assert(tstr == "dyn_concrete");

			// 
			p[0] = read_point();
			p[1] = read_point();
			p[2] = read_point();

			auto pathVertexBegin = vg.vertexNumber();

			int contourVertexBegin = pathVertexBegin;
			bool closed = false;

			//
			while (fin >> tstr) {
				if (tstr != "fL" && tstr.length() != 1) { break; }

				char opChar = tstr[0];
				if (string("MmZzLlHhVvCcSsQqTtAa").find(opChar) == string::npos) {
					if (tstr != "fL"){ break; }
				}

				switch (opChar) {
				case 'M':
					//if (!contour.empty()) { contour = vg.addContour(); }
					vg.addContour();
					p[0] = read_point();
					contourVertexBegin = vg.vertexNumber();
					closed = false;
					break;
				case 'L':
				{
					p[1] = read_point();
					vg.addCurve(CT_Linear);
					auto vi = vg.addVertex(2);
					vg.vertex.point[vi] = p[0];
					vg.vertex.point[vi + 1] = p[1];
					vg.cutToLength();
					p[0] = p[1];
					break;
				}
				case 'Q':
				{
					p[1] = read_point();
					p[2] = read_point();
					refPoint = p[1];
					vg.addCurve(CT_Quadratic);
					auto vi = vg.addVertex(3);
					vg.vertex.point[vi] = p[0];
					vg.vertex.point[vi + 1] = p[1];
					vg.vertex.point[vi + 2] = p[2];
					vg.reduceDegenerate();
					vg.cutToLength();
					p[0] = p[2];
					break;
				}
				case 'C':
				{
					p[1] = read_point();
					p[2] = read_point();
					p[3] = read_point();
					refPoint = p[2];
					vg.addCurve(CT_Cubic);
					auto vi = vg.addVertex(4);
					vg.vertex.point[vi] = p[0];
					vg.vertex.point[vi + 1] = p[1];
					vg.vertex.point[vi + 2] = p[2];
					vg.vertex.point[vi + 3] = p[3];
					vg.reduceDegenerate();
					vg.cutToLength();
					p[0] = p[3];
					break;
				}
				case 'A':
				{
					//QM: arcto - a vertex plus a w
					fin >> tstr;
					replace_comma(tstr);
					replace_colon(tstr);
					vec2 v_mid;
					float w1=0.f;
					sscanf(tstr.c_str(), "%f %f %f",&v_mid.x,&v_mid.y,&w1);
					if(!(w1>=0.f||w1<=0.f)){
						w1=0.f;
					}
					p[1] = read_point();
					if(w1==0.f){
						//half-circle, we need to break it down into two curves, here v_mid is a *vector* flag indicating the orientation
						vec2 delta=0.5f*(p[1]-p[0]);
						//vec2 dir_arc_mid=0.5f*vec2(delta.y,-delta.x);
						//if(dot(dir_arc_mid,v_mid)<0.f){dir_arc_mid=-1.f*dir_arc_mid;}
						//dir_arc_mid=v_mid;
						v_mid=0.5f*(p[0]+p[1])+v_mid;
						vec2 v_helper0=v_mid-delta;
						vec2 v_helper1=v_mid+delta;
						w1=0.7071067811865475244f;

						auto ci = vg.addCurve(CT_Rational);
						auto vi = vg.addVertex(3);

						vg.vertex.point[vi] = p[0];
						vg.vertex.point[vi + 1] = v_helper0;
						vg.vertex.point[vi + 2] = v_mid;
						vg.curve.arc_w1s[ci] = __float_as_int(w1);
						vg.cutToLength();

						ci = vg.addCurve(CT_Rational);
						vi = vg.addVertex(3);

						vg.vertex.point[vi] = v_mid;
						vg.vertex.point[vi + 1] = v_helper1;
						vg.vertex.point[vi + 2] = p[1];
						vg.curve.arc_w1s[ci] = __float_as_int(w1);
						vg.cutToLength();

						p[0] = p[1];
					}else{
						auto ci = vg.addCurve(CT_Rational);
						auto vi = vg.addVertex(3);

						vg.vertex.point[vi] = p[0];
						vg.vertex.point[vi + 1] = v_mid / w1;
						vg.vertex.point[vi + 2] = p[1];
						vg.curve.arc_w1s[ci] = __float_as_int(w1);
						vg.cutToLength();

						p[0] = p[1];
					}
					break;
				}
				case 'Z':
				{
					closed = true;
					break;
				}
				default:
					if (tstr == "fL") {
						//if (!contour.empty()) {
						//	path.c.push_back(contour);
						//	contour = RawContour();
						//}
						p[0] = read_point();
						break;
					}
					else {
						throw runtime_error("RVG::load(): unsupported comand");
						assert(0);
					}
					break;
				}

				lastOpChar = opChar;
			}

			if (vg.vertexNumber() == pathVertexBegin) {
				getline(fin, tstr);
				continue;
			}

			// close path
			if (!closed && vg.vertex.point.size() > contourVertexBegin) {
				// add a straight line.
				auto v_first = vg.vertex.point[contourVertexBegin];
				auto v_last = vg.vertex.point.back();

				if (v_first != v_last) {
					vg.addCurve(CT_Linear);
					auto vi = vg.addVertex(2);
					vg.vertex.point[vi] = v_last;
					vg.vertex.point[vi + 1] = v_first;
					vg.cutToLength();
				}

			}

			//
			glm::mat3x3 mat;
			if (tstr.substr(0, 10) == "dyn_affine") {

				tstr = tstr.substr(12);
				tstr.erase(tstr.find(']'), 1);
				tstr.erase(tstr.find('['), 1);
				tstr.erase(tstr.find(']'), 1);
				tstr = tstr.substr(0, tstr.length() - 1);
				
				replace_comma(tstr);
				std::istringstream iss(tstr);
				iss >> mat[0][0] >>  mat[1][0] >>  mat[2][0] 
					>> mat[0][1] >>  mat[1][1] >>  mat[2][1];
			}
			else if (tstr == "dyn_identity") {
			}
			else {
				assert(0);
			}

			uint32_t pathVertexEnd = vg.vertexNumber();

			auto tpoint = [&](const vec2 &v) -> vec2 {
				glm::vec3 v3 = mat * glm::vec3(v, 1.f);
				auto nv2 = vec2(v3.x, v3.y) / v3.z;
				//nv2.y = -nv2.y + this->_viewport[1].y;
				return nv2;
			};
			
			for (uint32_t i = pathVertexBegin; i < pathVertexEnd; ++i) {
				vg.vertex.point[i] = tpoint(vg.vertex.point[i]);
			}

			//
			fin >> tstr;
			assert(tstr == "dyn_paint");

			//
			float opacity;
			fin >> opacity;

			vg.path.fillOpacity[path] = opacity;

			//
			TGradientItem gi;
			memset(&gi,0,sizeof(gi));
			fin >> tstr;
			if (tstr == "solid") {

				frgba fc;
				fin >> tstr;

				if (tstr.substr(0, 4) == "rgba") {
					tstr = tstr.substr(5, tstr.length() - 6);
					replace_comma(tstr);
					std::istringstream iss(tstr);
					iss >> fc.r >> fc.g >> fc.b >> fc.a;
				}
				else if (tstr.substr(0, 3) == "rgb") {
					tstr = tstr.substr(4, tstr.length() - 6);
					replace_comma(tstr);
					std::istringstream iss(tstr);
					iss >> fc.r >> fc.g >> fc.b;
					fc.a = 1.f;
				}
				else {
					throw std::runtime_error("RVG::load() unknown color type");
				}

				u8rgba c;
				c = fc;

				vg.path.fillType[path] = FT_COLOR;
				c.a = (uint8_t)(c.a*opacity);

				if (_a128) {
					c.a = (uint8_t)((c.a / 255.f) * 127.f);
					c.a = (uint8_t)((c.a / 127.f) * 255.f);
				}

				//if (c.a < 16) { c.a &= 0xFE; }
				//if (c.a == 7) {
				//	printf("666\n");
				//	c.a = 6;
				//}

				vg.path.fillColor[path] = c;
				vg.path.fillIndex[path] = -1;

				//we need an empty gradient slot to simplify texture addressing
				temp_grad_items.push_back(gi);
			}
			else {
				// gradient;
				int ft = GT_Linear;
				if (tstr == "radial") { ft = GT_Radial; }
				else { ft = GT_Linear; assert(tstr == "linear"); }

				Gradient gradient;
				gradient.gradient_type = (GradientType)ft;

				///////
				getline(fin, tstr);
				int ppad = (int)tstr.find("pad");
				int n = (int)tstr.length();
				for (int i = 0; i < n; i++) {
					int ch = (int)(uint8_t)tstr[i];
					if ((uint32_t)(ch - 48) >= 10u && ch != '-'&&ch != '.'&&ch != '+'&&!(ch == 'e'&&i < n - 1 && (tstr[i + 1] == '+' || tstr[i + 1] == '-'))) {
						tstr[i] = ' ';
					}
				}

				std::string s0 = tstr.substr(0, ppad);
				std::string s1 = tstr.substr(ppad, n - ppad);
				std::stringstream s0in(s0);
				std::stringstream s1in(s1);

				//generate the transform
				//we need to multiply an inverted tr, then normalize it
				/*
				A   0    t0 t3 0         1/A  0
					   = t1 t4 0  ->
				b   1    t2 t5 1     - b 1/A  1
				*/
				float tr[6] = { 1.f, 0.f, 0.f, 0.f, 1.f, 0.f };
				float trinv[6] = { 1.f, 0.f, 0.f, 0.f, 1.f, 0.f };
				float focal_point[3] = { 0.f, 0.f, 0.f };

				s1in >> tr[0] >> tr[1] >> tr[2] >> tr[3] >> tr[4] >> tr[5];

				gradient.gradient_transform = glm::mat3x3(
					tr[0], tr[1], tr[2],
					tr[3], tr[4], tr[5],
					0, 0, 1
					);

				float det = (float)((double)tr[0] * (double)tr[4] - (double)tr[1] * (double)tr[3]);
				float idet = safeRcp(det);
				trinv[0] = tr[4] * idet; trinv[3] = -tr[3] * idet;
				trinv[1] = -tr[1] * idet; trinv[4] = tr[0] * idet;
				trinv[2] = -(float)((double)tr[2] * (double)trinv[0] + (double)tr[5] * (double)trinv[1]);
				trinv[5] = -(float)((double)tr[2] * (double)trinv[3] + (double)tr[5] * (double)trinv[4]);

				//printf("-------------------------\n");
				//printf("%f %f %f\n%f %f %f\n",tr[0],tr[1],tr[2],tr[3],tr[4],tr[5]);
				//printf("--\n");
				//printf("%f %f %f\n%f %f %f\n",trinv[0],trinv[1],trinv[2],trinv[3],trinv[4],trinv[5]);
				//memcpy(trinv,tr,sizeof(tr));

				///////////////////
				float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f, r = 0.f;
				float knot = 0.f, value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
				s0in >> x0 >> y0 >> x1 >> y1;
				trinv[2] -= x0;
				trinv[5] -= y0;
				if (ft == GT_Radial) {
					//subtract and scale
					s0in >> r;

					gradient.c = glm::vec2(x0, y0);
					gradient.f = glm::vec2(x1, y1);
					gradient.r = r;

					float ir = safeRcp(r);
					trinv[0] *= ir; trinv[1] *= ir; trinv[2] *= ir;
					trinv[3] *= ir; trinv[4] *= ir; trinv[5] *= ir;
					focal_point[0] = (x1 - x0)*ir;
					focal_point[1] = (y1 - y0)*ir;
				}
				else {
					//subtract and inverse-dot, leave 3,4,5 empty

					gradient.v1 = glm::vec2(x0, y0);
					gradient.v2 = glm::vec2(x1, y1);

					float dx = x1 - x0;
					float dy = y1 - y0;
					float _2ir2 = 2.f / (dx*dx + dy*dy);
					dx *= _2ir2; dy *= _2ir2;
					trinv[0] = trinv[0] * dx + trinv[3] * dy;
					trinv[1] = trinv[1] * dx + trinv[4] * dy;
					trinv[2] = trinv[2] * dx + trinv[5] * dy;
					trinv[3] = 0.f;
					trinv[4] = 0.f;
					trinv[5] = 0.f;
					trinv[2] -= 1.f;
					focal_point[0] = -1.f;
					focal_point[1] = 0.f;
				}
				focal_point[2] = 1.f - focal_point[0] * focal_point[0] - focal_point[1] * focal_point[1];
				if (!(focal_point[2] > 0.f)) { focal_point[2] = 0.f; }

				//generate the ramp
				temp_knots.clear();
				while (s0in >> knot) {
					s0in >> value0 >> value1 >> value2 >> value3;

					GradientStop stop;
					stop.offset = knot;
					stop.color.r = (uint8_t)(value0 * 255);
					stop.color.g = (uint8_t)(value1 * 255);
					stop.color.b = (uint8_t)(value2 * 255);
					stop.color.a = (uint8_t)(value3 * 255);

					gradient.gradient_stops.push_back(stop);

					temp_knots.push_back(knot);
					temp_knots.push_back(value0);
					temp_knots.push_back(value1);
					temp_knots.push_back(value2);
					temp_knots.push_back(value3*opacity);
				}
				if (!temp_knots.size()) {
					temp_knots.push_back(0.f);
					temp_knots.push_back(0.f);
					temp_knots.push_back(0.f);
					temp_knots.push_back(0.f);
					temp_knots.push_back(0.f);
				}
				if ((int)temp_knots.size() <= 5) {
					float val = temp_knots[0];
					temp_knots.resize(10);
					memcpy(&temp_knots[5], &temp_knots[0], sizeof(float) * 5);
					temp_knots[0] = 0.f;
					temp_knots[5] = 1.f;
				}
				float min_delta = 1.f;
				for (int i = 5; i < (int)temp_knots.size(); i += 5) {
					float delta = temp_knots[i] - temp_knots[i - 5];
					if (delta > 0.f&&delta < min_delta) { min_delta = delta; }
				}
				int res = (int)(16.f / min_delta);
				if (res < 1) { res = 1; }
				if (res > 1023) { res = 1023; }
				res |= res >> 1;
				res |= res >> 2;
				res |= res >> 4;
				res |= res >> 8;
				res++;
				memcpy(gi.tr, trinv, sizeof(trinv));
				memcpy(gi.focal_point, focal_point, sizeof(focal_point));
				gi.mode = ft;
				gi.p = (int)temp_ramp_buffer.size();
				gi.n = res;
				temp_grad_items.push_back(gi);
				float ires = 1.f / (float)(res - 1);
				int p_knot = 0;
				for (int i = 0; i < res; i++) {
					float t_i = (float)i*ires;
					float t0 = 0.f, t1 = 1.f;
					while (p_knot + 10 < (int)temp_knots.size()) {
						t0 = temp_knots[p_knot];
						t1 = temp_knots[p_knot + 5];
						if (t_i > t1 || t0 == t1) {
							p_knot += 5;
						}
						else {
							break;
						}
					}

					t0 = temp_knots[p_knot];
					t1 = temp_knots[p_knot + 5];
					float t_local = (t_i - t0)*safeRcp(t1 - t0);
					if (!(t_local > 0.f)) { t_local = 0.f; }
					if (t_local > 1.f) { t_local = 1.f; }

					int rgba;

					if (Mochimazui::VGConfig::sRGB()) {

						frgba sc0 = frgba(temp_knots[p_knot + 1], temp_knots[p_knot + 2], temp_knots[p_knot + 3], temp_knots[p_knot + 4]);
						frgba sc1 = frgba(temp_knots[p_knot + 6], temp_knots[p_knot + 7], temp_knots[p_knot + 8], temp_knots[p_knot + 9]);

						auto lc0 = srgb_to_lrgb(sc0);
						auto lc1 = srgb_to_lrgb(sc1);

						auto ss = lerp(sc0, sc1, t_local);
						auto sl = srgb_to_lrgb(ss);

						auto ll = lerp(lc0, lc1, t_local);
						auto ls = lrgb_to_srgb(ll);

						auto rc = ls;

						rgba = f32tou8(rc.r) << (8 * 0);
						rgba += f32tou8(rc.g) << (8 * 1);
						rgba += f32tou8(rc.b) << (8 * 2);
					}
					else {
						rgba = f32tou8(lerp(temp_knots[p_knot + 1], temp_knots[p_knot + 6], t_local)) << (8 * 0);
						rgba += f32tou8(lerp(temp_knots[p_knot + 2], temp_knots[p_knot + 7], t_local)) << (8 * 1);
						rgba += f32tou8(lerp(temp_knots[p_knot + 3], temp_knots[p_knot + 8], t_local)) << (8 * 2);
					}

					// alpha
					rgba += f32tou8(lerp(temp_knots[p_knot + 4], temp_knots[p_knot + 9], t_local)) << (8 * 3);

					temp_ramp_buffer.push_back(rgba);
				}

				u8rgba c;
				c.r = 255; c.g = 0; c.b = 255; c.a = 0;
				vg.path.fillType[path] = FT_GRADIENT;
				vg.path.fillColor[path] = c;
				vg.path.fillIndex[path] = -1;

				vg.gradient.svg_gradients.push_back(gradient);
				vg.path.fillIndex[path] = (int)vg.gradient.svg_gradients.size() - 1;
			}

			if (!fin) {
				throw runtime_error("RVG::load(): error while reading.");
				assert(0);
			}
		}

		{
			// post-process the gradients
			// repack every thing in n descending order: 
			//     we are guaranteed to get interpolatable packing intervals

			std::vector<int> _sort_index;
			int n = (int)temp_grad_items.size();
			_sort_index.resize(n);
			for (int i = 0; i < n; i++) { _sort_index[i] = i; }
			auto by_n_descending = [&temp_grad_items](int a, int b) {
				return temp_grad_items[a].n > temp_grad_items[b].n;
			};
			std::sort(_sort_index.begin(), _sort_index.end(), by_n_descending);

			//
			int h_ramp_texture = ((int)temp_ramp_buffer.size() + 1023) / 1024;
			vg.gradient.gradient_ramp_texture.resize(h_ramp_texture * 1024);

			int pout = 0;
			for (int i = 0; i < n; i++) {
				int id = _sort_index[i];
				int p0 = temp_grad_items[id].p;
				temp_grad_items[id].p = pout;
				if (temp_grad_items[id].n) {
					memcpy(&vg.gradient.gradient_ramp_texture[pout],
						&temp_ramp_buffer[p0],
						temp_grad_items[id].n*sizeof(int));
					pout += temp_grad_items[id].n;
				}
			}
			vg.gradient.gradient_table_texture.resize(n * 12);
			for (int i = 0; i < n; i++) {
				//temp_grad_items[i].n=temp_grad_items[i].n*2+(temp_grad_items[i].mode==FT_RADIAL_GRADIENT);
				//GPU doesn't need to know the mode - they are identical after transformation providing you don't clamp the linear gradient to 0
				memcpy(&vg.gradient.gradient_table_texture[i * 12], &temp_grad_items[i], 12 * sizeof(int));
				*(float*)&vg.gradient.gradient_table_texture[i * 12 + 7] = (float)(temp_grad_items[i].n - 1)*(1.f / 1024.f);
			}
			if (!vg.gradient.gradient_table_texture.size()) {
				vg.gradient.gradient_table_texture.resize(12);
				memset(&vg.gradient.gradient_table_texture[0], 0, 12 * sizeof(int));
			}
		}

		//vg.updatePathDirection();

		vg.width = this->width();
		vg.height = this->height();

		// only appeared in reschart.rvg
		//if (_viewport[0].x != 0) {
		//	for (auto &v : vg.vertex.point) {
		//		v.x -= _viewport[0].x;
		//	}
		//}

		//if (Mochimazui_Old_ConfigVerbose()) {
		//	cout << "RVG::load() end" << endl;
		//}
	}

	void RVG::saveSelectedPath(const std::vector<uint32_t> &pids) {

		if (!pids.size()) {
			return;
		}

		std::string ofname;

		for (int i = 1;;++i) {
			std::ostringstream oss;
			oss << "./test_rvg/selected/selected_" << i << ".rvg";

			ifstream fin;
			fin.open(oss.str());
			if (!fin) { 
				ofname = oss.str();
				break;
			}
		}

		std::ofstream fout;
		fout.open(ofname);

		fout << _header;

		for (auto id : pids) {
			fout << _lines[id] << endl;
		}	
	}
}


