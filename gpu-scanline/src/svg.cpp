
#define _CRT_SECURE_NO_WARNINGS

#include "svg.h"

#include "rapidxml.hpp"
#include "rapidxml_print.hpp"

#include <mochimazui/stdio_ext.h>
#include <mochimazui/bitmap.h>

//#include <opencv2/core/core.hpp>
//#include <opencv2/core/mat.hpp>
//#include <opencv2/core/matx.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

namespace Mochimazui {

using std::shared_ptr;
using stdext::string;

void SVG::setVg(shared_ptr<VGContainer> &p_vg) {
	_spVGContainer = p_vg;
}

void SVG::save(const string &file_name) {

	printf("SVG::save\n");

	FILE *fout;
	fout = fopen(file_name.c_str(), "wb");

	if (!fout) {
		stdext::error_printf("SVG::save: cannot open file \"%s\"\n", file_name.c_str());
		return;
	}

	if (!_spVGContainer) { 
		stdext::error_printf("SVG::save: empty vg\n");
		fclose(fout);
		return; 
	}

	auto &vg = *_spVGContainer;

	fprintf(fout, "<?xml version=\"1.0\" standalone=\"no\"?>\n");
	fprintf(fout, "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" xmlns:svg=\"http://www.w3.org/2000/svg\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" viewBox=\"0 0 %d %d\">\n",
		_spVGContainer->width, _spVGContainer->height);

	//node->append_attribute(doc.allocate_attribute("viewBox", vb_str));

	auto v2s = [](const glm::vec2 &p) {
		char s[64] = { 0 };
		sprintf(s, "%f %f ", p.x, p.y);
		return std::string(s);
	};

	for (int i = 0; i < vg.gradient.svg_gradients.size(); ++i) {
		auto &g = vg.gradient.svg_gradients[i];

		if (g.gradient_type == GT_Linear) {
			fprintf(fout, R"o(<linearGradient id="G%d" x1="%f" y1="%f" x2="%f" y2="%f" spreadMethod="pad" gradientUnits="userSpaceOnUse" gradientTransform="matrix(%f,%f,%f,%f,%f,%f)">)o""\n",
				i, g.v1.x, g.v1.y, g.v2.x, g.v2.y,
				g.gradient_transform[0][0], g.gradient_transform[1][0],
				g.gradient_transform[0][1], g.gradient_transform[1][1],
				g.gradient_transform[0][2], g.gradient_transform[1][2]
				);
		}
		else {
			fprintf(fout, R"o(<radialGradient id="G%d" cx="%f" cy="%f" fx="%f" fy="%f" r="%f" spreadMethod="pad" gradientUnits="userSpaceOnUse" gradientTransform="matrix(%f,%f,%f,%f,%f,%f)">)o""\n",
				i,g.c.x, g.c.y, g.f.x, g.f.y, g.r,
				g.gradient_transform[0][0], g.gradient_transform[1][0],
				g.gradient_transform[0][1], g.gradient_transform[1][1],
				g.gradient_transform[0][2], g.gradient_transform[1][2]
				);
		}

		for (int j = 0; j < g.gradient_stops.size(); ++j) {
			auto &s = g.gradient_stops[j];
			fprintf(fout, "\t" R"o(<stop offset="%f" stop-color="rgb(%d,%d,%d)" stop-opacity="%f"/>)o""\n",
				s.offset, s.color.r, s.color.g, s.color.b, s.color.a / 255.f);
		}

		if (g.gradient_type == GT_Linear) {
			fprintf(fout, "</linearGradient>\n");
		}
		else {
			fprintf(fout, "</radialGradient>\n");
		}

	}

	fprintf(fout, "<g transform = \"scale(1,-1) translate(0,-%d)\">\n", _spVGContainer->height);

	std::string d;

	//
	for (uint32_t i = 0; i < vg.pathNumber(); ++i) {
		std::string fill_rule_str = vg.path.fillRule[i] ? "evenodd" : "nonzero";
		auto fill_color = vg.path.fillColor[i];
		char fill_str[32];
		float fill_opacity = 1.f;

		bool has_stroke = false;
		auto stroke_color = vg.path.strokeColor[i];
		char stroke_str[32];
		float stroke_width = vg.path.strokeWidth[i];
		float stroke_opacity = 1.f;

		if (vg.path.fillType[i] == FT_COLOR) {
			//sprintf(fill_str, "rgba(%d,%d,%d,%d)", fill_color.r, fill_color.g, fill_color.b,
			//	fill_color.a);

			sprintf(fill_str, "rgb(%d,%d,%d)", fill_color.r, fill_color.g, fill_color.b);
			fill_opacity = fill_color.a / 255.f;
		}
		else if (vg.path.fillType[i] == FT_NONE) {
			sprintf(fill_str, "none");
			fill_opacity = 1.f;
		}
		else
		{
			auto gid = vg.path.fillIndex[i];
			sprintf(fill_str, "url(#G%d)", gid);
			fill_opacity = vg.path.fillOpacity[i];
			uint8_t u8_opacity = (uint8_t)(fill_opacity * 255.f);
			fill_opacity = u8_opacity / 255.f;
		}

		if (vg.path.strokeType[i] == FT_NONE) {
			sprintf(stroke_str, "none");
		}
		else if (vg.path.strokeType[i] == FT_COLOR) {
			sprintf(stroke_str, "rgb(%d,%d,%d)", stroke_color.r, stroke_color.g, stroke_color.b);
			stroke_opacity = stroke_color.a / 255.f;
			has_stroke = true;
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
					d.push_back(' ');
					d += v2s(vg.vertex.point[vertex_pos]);
				}

				if (curve_type == CT_Linear) { d.push_back('L'); }
				else if (curve_type == CT_Quadratic) { d.push_back('Q'); }
				else if (curve_type == CT_Cubic) { d.push_back('C'); }
				else if (curve_type == CT_Rational) { 

					glm::vec2 iv0, iv1, iv2;

					iv0 = vg.vertex.point[vertex_pos];
					iv1 = vg.vertex.point[vertex_pos + 1];
					iv2 = vg.vertex.point[vertex_pos + 2];

					float iw = *((float*)&vg.curve.arc_w1s[curve_pos + k]);

					glm::vec2 raw_iv[4] = {
						iv0, iv1, iv2,
						glm::vec2(iw, 0.f)
					};

					iv1 *= iw;

					auto cp = [iv0, iv1, iv2, iw](float t) {
						glm::vec2 lv0 = iv0 * (1 - t) + iv1 * t;
						glm::vec2 lv1 = iv1 * (1 - t) + iv2 * t;
						glm::vec2 v = lv0 * (1 - t) + lv1 * t;
						float w = (1 - t)*(1 - t) + 2.f*(1 - t)*t*iw + t*t;
						auto safeRcp = [](float a) { return a ? (1.f / a) : 0.f; };
						v *= safeRcp(w);
						return v;
					};

					auto ov1 = cp(0.5f);

					enum ArcMode {
						ArcAsQuad,
						ArcToArc,
						ArcSubdiv,
					};

					//auto am = ArcToArc;
					auto am = ArcSubdiv;

					switch (am) {
					case ArcAsQuad:
					{
						d.push_back('Q');
						d.push_back(' ');
						d += v2s(raw_iv[1]);
						d += v2s(iv2);
						break;
					}
					case ArcToArc:
					{
						auto r2 = iv2 - iv0;
						auto r = sqrt(r2.x * r2.x + r2.y * r2.y);

						char cmd_a[128];
						sprintf(cmd_a, "%f,%f 0 0,0 %f,%f ",
							r, r, iv2.x, iv2.y);

						d += cmd_a;
						break;
					}
					case ArcSubdiv:
					{

						// -------- -------- -------- -------- 
						// -------- -------- -------- -------- 

						std::function<void(glm::vec2 *)> rational_curve_tes;
						rational_curve_tes = [&](glm::vec2 cv[4]) {

							glm::vec2 vfirst, vlast;
							vfirst = cv[0];
							vlast = cv[2];

							bool stop = false;
							glm::vec2 mid_p;

							{
								auto cp = [cv](float t) {
									glm::vec2 lv0 = cv[0] * (1 - t) + cv[1] * t;
									glm::vec2 lv1 = cv[1] * (1 - t) + cv[2] * t;
									glm::vec2 v = lv0 * (1 - t) + lv1 * t;
									auto iw = cv[3].x;
									float w = (1 - t)*(1 - t) + 2.f*(1 - t)*t*iw + t*t;
									auto safeRcp = [](float a) { return a ? (1.f / a) : 0.f; };
									v *= safeRcp(w);
									return v;
								};

								mid_p = cp(0.5f);

								static const float TLEN = 1.f / 16.f;
								if (abs(vfirst.x - vlast.x) < TLEN && abs(vfirst.y - vlast.y) < TLEN) {
									stop = true;
								}
								else {

									auto v0 = vfirst;
									auto v1 = mid_p;
									auto v2 = vlast;

									auto d = glm::normalize(v2 - v0);
									auto b = glm::dot(v1 - v0, d);
									auto c = sqrt(glm::dot(v1 - v0, v1 - v0));
									auto a = sqrt(c*c - b*b);

									if (a < 1.f / 64.f) {
										stop = true;
									}

								}

							}

							if(stop) {

								// use quad ?
								auto v0 = vfirst;
								glm::vec2 v1;
								auto v2 = vlast;

								char cmd[128];

								sprintf(cmd, "L %f %f ", vlast.x, vlast.y);
								//sprintf(cmd, "Q %f,%f %f,%f ", v1.x, v1.y, v2.x, v2.y);
								d += cmd;

								return;
							}

							{
								glm::vec2 out[4];

								auto blossom = [](glm::vec2 *B, float Bw, float u, float v, float &w) -> glm::vec2
								{
									float uv = u*v;
									float b0 = uv - u - v + 1,
										b1 = u + v - 2 * uv,
										b2 = uv;

									w = 1 * b0 + Bw*b1 + 1 * b2;

									return B[0] * b0 + B[1] * b1 + B[2] * b2;
								};

								auto subcurve = [&](float u, float v) {

									glm::vec2 cB[3] = { cv[0], cv[1], cv[2] };
									float cBw = cv[3].x;

									float wA, wB, wC;
									glm::vec2 A = blossom(cB, cBw, u, u, wA);
									glm::vec2 B = blossom(cB, cBw, u, v, wB);
									glm::vec2 C = blossom(cB, cBw, v, v, wC);

									float s = 1.0f / sqrt(wA * wC);
									out[1] = s*B;
									//out.w = s*wB;
									out[3].x = s*wB;

									if (u == 0)
									{
										out[0] = cB[0];
										out[2] = C / wC;
									}
									else if (v == 1)
									{
										out[0] = A / wA;
										out[2] = cB[2];
									}
									else
									{
										out[0] = A / wA;
										out[2] = C / wC;
									}
									//return out;
								};

								subcurve(0.f, 0.5f);
								rational_curve_tes(out);

								subcurve(0.5f, 1.f);
								rational_curve_tes(out);
							}

						};

						raw_iv[1] *= iw;
						rational_curve_tes(raw_iv);

						break;

						// -------- -------- -------- -------- 
						// -------- -------- -------- -------- 

					}
					}
				}
				else { throw std::runtime_error("SVG::save illegal curve type."); }

				d.push_back(' ');
				if (curve_type != CT_Rational) {
					for (int l = 1; l < (curve_type & 7); ++l) {
						d += v2s(vg.vertex.point[vertex_pos + l]);
					}
				}

			}
		}

		if (!has_stroke) {
			d.push_back('Z');
		}
		d.push_back(' ');

		if (has_stroke) {
			fprintf(fout, "  <path fill-rule=\"%s\" fill=\"%s\" fill-opacity=\"%f\" stroke=\"%s\" stroke-width=\"%f\" stroke-opacity=\"%f\" color-interpolation=\"linearRGB\" d=\"%s\"/>\n",
				fill_rule_str.c_str(), fill_str, fill_opacity, stroke_str, stroke_width, stroke_opacity, d.c_str());
		}
		else {
			fprintf(fout, "  <path fill-rule=\"%s\" fill=\"%s\" fill-opacity=\"%f\" color-interpolation=\"linearRGB\" d=\"%s\"/>\n",
				fill_rule_str.c_str(), fill_str, fill_opacity, d.c_str());
		}
		

		//fprintf(fout, "  <path fill-rule=\"%s\" fill=\"%s\" fill-opacity=\"%f\" color-interpolation=\"sRGB\" d=\"%s\"/>\n",
		//	fill_rule_str.c_str(), fill_str, fill_opacity, d.c_str());

		//fprintf(fout, "  <path fill-rule=\"%s\" fill=\"%s\" fill-opacity=\"%f\" d=\"%s\"/>\n",
		//	fill_rule_str.c_str(), fill_str, fill_opacity, d.c_str());

	}

	fprintf(fout, "</g>\n");
	fprintf(fout, "</svg>\n");

	//std::vector<char> out_vector;
	//std::back_insert_iterator<std::vector<char>> oi(out_vector);
	//rapidxml::print(oi, doc);

	//fwrite(out_vector.data(), 1, out_vector.size(), fout);

	fclose(fout);
}

}