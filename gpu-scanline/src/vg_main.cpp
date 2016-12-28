
#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>

#include <map>
#include <memory>
#include <string>
#include <algorithm>

#include <boost/program_options.hpp>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <cuda_profiler_api.h>

#include <mochimazui/glgui.h>
#include <mochimazui/string.h>
#include <mochimazui/stdio_ext.h>
#include <mochimazui/camera_controller_2d.h>
#include <mochimazui/camera_controller_3d.h>

#include <thrust/version.h>

#include "rvg.h"
#include "svg.h"
#include "vg_config.h"
#include "vg_container.h"
#include "timer.h"

#include "rasterizer/shared/ras_base.h"
#include "rasterizer/shared/ras_factory.h"

// -------- -------- -------- -------- -------- -------- -------- --------
using std::shared_ptr;
using std::weak_ptr;

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4x4;

namespace GLGUI = Mochimazui::GLGUI;
using GLGUI::Application;
using GLGUI::SubWindow;

using Mochimazui::CameraController2D;
using Mochimazui::CameraController3D;
//using Mochimazui_Old_Config::Config;
namespace Config = Mochimazui::VGConfig;
using Mochimazui::VGContainer;
using Mochimazui::SVG;
using Mochimazui::RVG;

using Mochimazui::stdext::color_printf;
using Mochimazui::stdext::error_printf;

// -------- -------- -------- -------- -------- -------- -------- --------
enum GLUTActiveTool {
	AT_NULL,
	AT_MOVE,
	AT_TURN,
	AT_ROTATE,
	AT_PEN,
	AT_BRUSH,
};

// -------- -------- -------- -------- -------- -------- -------- --------

std::shared_ptr<Application> g_app;

struct {
	shared_ptr<GLGUI::Window> window;
	weak_ptr<GLGUI::SubWindow> displayWindow;
} g_ui;

bool g_leftButtonDown = false;
bool g_rightButtonDown = false;
glm::ivec2 g_mousePos;

// -------- -------- -------- -------- -------- -------- -------- --------
glm::ivec2 g_window_pos;
glm::ivec2 g_window_size;

glm::ivec2 g_output_size;

glm::vec2 g_vg_size;

// -------- -------- -------- -------- -------- -------- -------- --------
CameraController3D g_inputController;
CameraController3D g_outputController;

std::shared_ptr<Mochimazui::RasterizerBase::VGRasterizer> g_rasterizer;

// -------- -------- -------- -------- -------- -------- -------- --------
// pen & brush

int g_activatedTool = AT_PEN;
int g_activatedTool_bk = AT_PEN;

uint32_t g_penColor = 0xFF000000;
uint32_t g_brushColor = 0xC0CCCCCC;
float g_brush_size = 16.f;
float g_text_size = 128;

bool g_newPath = false;
bool g_firstPenSeg = true;

vec2 g_lastScenePos2;
vec2 g_lastScenePos;

vec2 g_click_scene_pos;

// -------- -------- -------- -------- -------- -------- -------- --------
void updateRasterizerTransform() {

	auto itmat = g_inputController.modelViewMatrix();
	g_rasterizer->setInputTransform(itmat);
	g_rasterizer->setOutputTransform(g_outputController.modelViewMatrix());

	auto rtmat = glm::inverse(itmat);

	float W = (float)g_window_size.x;
	float H = (float)g_window_size.y;

	// ray
	glm::vec4 rp(W / 2, H / 2, -std::min(W, H), 1.f);

	// planerm w
	glm::vec4 pp(W / 2, H / 2, 0.f, 1.f);
	glm::vec4 pd(0.f, 0.f, -1.f, 0.f);

	//
	rp = rtmat * rp; //rp /= rp.w;
	glm::vec4 rx = rtmat * glm::vec4(1.f, 0.f, 0.f, 0.f);
	glm::vec4 ry = rtmat * glm::vec4(0.f, 1.f, 0.f, 0.f);
	glm::vec4 rw = rtmat * glm::vec4(0.f, 0.f, 0.f, 1.f) - rp;

	float a = (pp.z - rp.z);
	g_rasterizer->setRevTransform(rx, ry, rw, rp, a);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void benchmark() {

	const int prepare_count = 16;
	const int benchmark_count = 256;

	printf("Benchmark: prepare\n");
	for (int i = 0; i < prepare_count; ++i) {
		g_rasterizer->rasterize();
	}

	auto count_pixel = Config::CountPixel();
	if (count_pixel) {
		printf("Count pixel.\n");
		auto pixel_count = g_rasterizer->pixelCount();
		printf("Pixel count: %d\n", pixel_count);
		auto attach_to_file = Config::AttachPixelCountToFile();
		FILE *fout = fopen(attach_to_file.c_str(), "a");
		if (fout) {
			auto osize = g_rasterizer->outputBufferSize();
			fprintf(fout, "%s,vg2-%s,%s%d%s,%dx%d,%llu,\n",
				Config::Name().c_str(),
				Config::sRGB() ? "sRGB" :	"linear-RGB",
				Config::MultisampleOutput() ? "ms" : "",
				Config::Samples(),
				"",
				osize.x, osize.y,
				pixel_count
				);
			fclose(fout);
		}
		exit(0);
	}

	printf("Benchmark: start\n");
	//Mochimazui::Timer benchmark_timer;

	LARGE_INTEGER pc_before;
	LARGE_INTEGER pc_after;

	LARGE_INTEGER pc_frequency;

	QueryPerformanceFrequency(&pc_frequency);

	double min_tpf = FLT_MAX;
	for (int i = 0; i < benchmark_count; ++i) {
		QueryPerformanceCounter(&pc_before);
		g_rasterizer->rasterize();
		QueryPerformanceCounter(&pc_after);

		double dtime =
			static_cast<double>(
				((*(long long *)&pc_after) - (*(long long *)&pc_before))
				);
		dtime /= *(long long *)&pc_frequency;

		min_tpf = std::min(min_tpf, dtime);
	}

	printf(">>> min_time_per_frame %f\n", min_tpf * 1000);
	printf(">>> max_frame_per_second %f\n", 1.0 / min_tpf);

	QueryPerformanceCounter(&pc_before);
	for (int i = 0; i < benchmark_count; ++i) {		
		g_rasterizer->rasterize();
	}
	QueryPerformanceCounter(&pc_after);

	double dtime =
		static_cast<double>(
			((*(long long *)&pc_after) - (*(long long *)&pc_before))
			);
	dtime /= *(long long *)&pc_frequency;

	printf(">>> average_time_per_frame %f\n", dtime * 1000 / benchmark_count);
	printf(">>> average_frame_per_second %f\n", 1.0 / (dtime / benchmark_count) );

	auto attach = Config::AttachTimingToFile();

	FILE *fout = fopen(attach.c_str(), "a");
	if (!fout) { return; }

	auto osize = g_rasterizer->outputBufferSize();
	fprintf(fout, "%s,vg2-%s,%s%d%s,%dx%d,%f,%f,%f,%f\n",
		Config::Name().c_str(),
		Config::sRGB() ? "sRGB" : "linear-RGB",
		Config::MultisampleOutput() ? "ms" : "",
		Config::Samples(),
		"",
		osize.x, osize.y,
		1.0 / min_tpf,
		min_tpf * 1000,
		1.0 / (dtime / benchmark_count),
		dtime * 1000 / benchmark_count);

	fclose(fout);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void step_timing() {

	const int prepare_count = 16;
	const int benchmark_count = 128;

	printf("Step timing: prepare\n");
	for (int i = 0; i < prepare_count; ++i) {
		g_rasterizer->rasterize();
	}

	printf("Step timing: start\n");

	std::vector<double> step_timing;
	step_timing = g_rasterizer->stepTiming();

	for (int i = 0; i < benchmark_count; ++i) {
		g_rasterizer->rasterize();
		auto nst = g_rasterizer->stepTiming();
		if (nst.size() != step_timing.size()) {
			throw std::runtime_error("incorrect step timing");
		}
		for (int k = 0; k < nst.size(); ++k) {
			step_timing[k] = std::min(step_timing[k], nst[k]);
		}
	}

	auto attach = Config::AttachTimingToFile();

	FILE *fout = fopen(attach.c_str(), "a");
	if (!fout) { return; }

	auto osize = g_rasterizer->outputBufferSize();
	auto nums = g_rasterizer->elementNumber();

	fprintf(fout, "%s,vg2-%s,%s%d,%dx%d,",
		Config::Name().c_str(),
		Config::sRGB() ? "sRGB" : "linear-RGB",
		Config::MultisampleOutput() ? "ms" : "",
		Config::Samples(),
		osize.x, osize.y
		);

	for (int i = 0; i < step_timing.size(); ++i) {
		fprintf(fout, "%f,", step_timing[i] * 1000);
	}

	{
		fprintf(fout, "*,%d,%d,%d,%d,%d,%d,",
			nums.path,
			nums.curve,
			nums.vertex,
			nums.curve_fragment,
			nums.merged_fragment,
			nums.span
		);
	}

	fprintf(fout, "\n");

	fclose(fout);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void onPaint() {
	g_rasterizer->rasterize();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void onResize(int w, int h) {

	if (Config::Verbose()) {
		printf("Resize: (%d,%d)\n", w, h);
	}

	auto pw = g_ui.displayWindow.lock();
	auto pos = pw->position();
	auto size = pw->size();
	g_window_pos = glm::ivec2(pos.x, pos.y);
	g_window_size = glm::ivec2(size.x, size.y);

	g_rasterizer->viewport(pos.x, pos.y, size.x, size.y);
	if (!Config::FixOutputSize()) {
		g_output_size = g_window_size;
		g_rasterizer->resizeOutputBuffer(size.x, size.y);
	}

	g_inputController.init((int)g_vg_size.x, (int)g_vg_size.y);
	if (Config::FitVGToWindowSize()) {
		g_inputController.fitToView(g_output_size.x, g_output_size.y);
	}	
	updateRasterizerTransform();
}

// -------- -------- -------- -------- -------- -------- -------- --------
vec3 glut_scene_rev_proj(const glm::vec2& p) {

	using glm::dvec3;
	using glm::dvec4;
	using glm::dmat4x4;

	using glm::vec3;
	using glm::vec4;
	using glm::mat4x4;

	float W = (float)g_window_size.x;
	float H = (float)g_window_size.y;

	dmat4x4 tmat = g_inputController.modelViewMatrix();
	auto rtmat = glm::inverse(tmat);

	// ray
	dvec4 rp(W / 2, H / 2, -std::min(W, H), 1.f);
	dvec4 rd = dvec4(p.x, p.y, 0, 1.f) - rp;

	// planerm w
	dvec4 pp(W / 2, H / 2, 0.f, 1.f);
	dvec4 pd(0.f, 0.f, -1.f, 0.f);

	//
	rp = rtmat * rp; rp /= rp.w;
	rd = rtmat * rd;

	double a = (rp.x - pp.x) * pd.x
		+ (rp.y - pp.y) * pd.y
		+ (rp.z - pp.z) * pd.z;

	double b = rd.x * pd.x
		+ rd.y * pd.y
		+ rd.z * pd.z;

	double t = -a / b;

	auto ip = rp + rd * t;
	return vec3(ip.x, ip.y, ip.z);
}

// -------- -------- -------- -------- -------- -------- -------- --------
vec2 glut_view_rev_proj(const vec2 &p) {
	auto dwSize = g_window_size;
	return vec2(p.x, dwSize.y - p.y);
}

void onLeftButtonDown(int x, int y) {
	g_newPath = true;

	// reverse transform.
	vec3 scenep = glut_scene_rev_proj(
		glut_view_rev_proj(vec2(x, y)));

	g_click_scene_pos = vec2(scenep.x, scenep.y);

	glm::ivec3 ip;
	ip.x = (int)floor(scenep.x);
	ip.y = (int)floor(scenep.y);

	// record position for pen & brush.
	g_lastScenePos2.x = scenep.x;
	g_lastScenePos2.y = scenep.y;
	g_lastScenePos.x = scenep.x;
	g_lastScenePos.y = scenep.y;
	g_firstPenSeg = true;

	// pass message to camera/scene controller.
	auto dwSize = g_window_size;
	g_leftButtonDown = true;
}

void onLeftButtonUp(int x, int y) {
	auto dwSize = g_window_size;
	g_leftButtonDown = false;
}

void onRightButtonDown(int x, int y) {

	auto state = SDL_GetKeyboardState(NULL);
	auto cflag = state[SDL_SCANCODE_RSHIFT] ? 1 : 0;
	if (cflag) {
		vec3 sp = glut_scene_rev_proj(
			glut_view_rev_proj(vec2(x, y)));
		g_rasterizer->startTextInput((int)sp.x, (int)sp.y, g_text_size, 0xFF000000);
	}

	auto dwSize = g_window_size;
	g_inputController.leftButtonDown(x, dwSize.y - 1 - y);
	g_outputController.leftButtonDown(x, dwSize.y - 1 - y);
	g_rightButtonDown = true;
}

void onRightButtonUp(int x, int y) {
	auto dwSize = g_window_size;
	g_inputController.leftButtonUp(x, dwSize.y - 1 - y);
	g_outputController.leftButtonUp(x, dwSize.y - 1 - y);
	g_rightButtonDown = false;
}

void onMouseMove(int x, int y, uint32_t buttonState) {

	g_mousePos.x = x;
	g_mousePos.y = y;

	// reverse transform.
	vec3 scenep = glut_scene_rev_proj(glut_view_rev_proj(vec2(x, y)));
#ifdef _DEBUG
	printf("mouse: (%.0f,%.0f)\n", scenep.x, scenep.y);
#endif
	
	if (!g_leftButtonDown && !g_rightButtonDown) { return; }

	// -------- -------- -------- --------
	auto state = SDL_GetKeyboardState(NULL);

	auto cflag = state[SDL_SCANCODE_LCTRL] ? 1 : 0;
	auto sflag = state[SDL_SCANCODE_LSHIFT] ? 1 : 0;

	//
	auto pController = &g_inputController;
	auto dwSize = g_window_size;

	if (cflag) {
		pController->setControllerMode(Mochimazui::CCM_TURN);
	}
	else if (sflag) {
		pController->setControllerMode(Mochimazui::CCM_ROTATE);
	}
	else {
		pController->setControllerMode(Mochimazui::CCM_MOVE);
	}

	if (g_rightButtonDown) {
		pController->move(x, dwSize.y - 1 - y);
	}

	glm::ivec3 ip;
	ip.x = (int)floor(scenep.x);
	ip.y = (int)floor(scenep.y);

	glm::vec2 currentScenePos;
	currentScenePos.x = scenep.x;
	currentScenePos.y = scenep.y;

	//updateRasterizerTransform();

	// pen & brush
	if (g_leftButtonDown&&glm::length(currentScenePos - g_lastScenePos) > 0.1f) {
		//if (g_activatedTool == AT_PEN) {
		if (!cflag) {
			auto lp2 = g_lastScenePos2;
			auto lp = g_lastScenePos;
			auto cp = currentScenePos;
			if (glm::length(cp - lp)<1.f) {
				//require a minimal movement
				return;
			}
			if (glm::length(lp - lp2)>0.1f) {
				auto N0 = lp - lp2; N0 = glm::normalize(glm::vec2(N0.y, -N0.x))*sqrt(g_brush_size / 40.f);
				auto N1 = cp - lp; N1 = glm::normalize(glm::vec2(N1.y, -N1.x))*sqrt(g_brush_size / 40.f);

				std::vector<float2> f2v;
				f2v.resize(8);
				glm::vec2 *v = (glm::vec2*)f2v.data();

				if (g_firstPenSeg) {
					g_firstPenSeg = false;
				}
				else {
					//hack: remove the previous segment
					//todo
				}

				v[0] = lp - N0;
				v[1] = lp + N0;

				v[2] = lp + N0;
				v[3] = cp + N1;

				v[4] = cp + N1;
				v[5] = cp - N1;

				v[6] = cp - N1;
				v[7] = lp - N0;

				std::vector<uint32_t> c;
				c.push_back(g_penColor);
				//g_rasterizer->addInk(f2v, c, g_newPath);
				g_rasterizer->addInk(f2v, c, true);
				g_newPath = false;
			}
		}
		else { //if (g_activatedTool == AT_BRUSH) {
			auto lp = g_lastScenePos;
			auto cp = currentScenePos;
			if (cp.x < lp.x) { std::swap(lp, cp); }
			float bh = g_brush_size*0.5f;

			std::vector<float2> f2v;
			f2v.resize(8);
			glm::vec2 *v = (glm::vec2*)f2v.data();

			v[0] = lp; v[0].y -= bh;
			v[1] = cp; v[1].y -= bh;

			v[2] = cp; v[2].y -= bh;
			v[3] = cp; v[3].y += bh;

			v[4] = cp; v[4].y += bh;
			v[5] = lp; v[5].y += bh;

			v[6] = lp; v[6].y += bh;
			v[7] = lp; v[7].y -= bh;

			std::vector<uint32_t> c;
			c.push_back(g_brushColor);
			//g_rasterizer->addInk(f2v, c, g_newPath);
			g_rasterizer->addInk(f2v, c, true);
			g_newPath = false;
		}

		g_lastScenePos2 = g_lastScenePos;
		g_lastScenePos.x = scenep.x;
		g_lastScenePos.y = scenep.y;
	}

	//
	updateRasterizerTransform();

}

void onMouseWheel(int x, int y) {

	static const float zin = 1.1f;
	static const float zout = 0.9f;

	auto state = SDL_GetKeyboardState(NULL);

	auto cflag = state[SDL_SCANCODE_LCTRL] ? 1 : 0;
	CameraController3D * pController = state[SDL_SCANCODE_LCTRL] ? &g_outputController : &g_inputController;

	auto mp = g_mousePos;
	mp.y = g_window_size.y - mp.y;

	if (y > 0) {
		// zoom in 
		pController->scale(glm::vec3(zin, zin, zin), glm::vec3(mp.x, mp.y, 0));	
	} else if (y < 0) {
		// zoom out
		pController->scale(glm::vec3(zout, zout, zout), glm::vec3(mp.x, mp.y, 0));
	} else {
		throw "";
	}

	if (cflag == 1) {
		g_rasterizer->enableOutputTransform(true);
	}
	updateRasterizerTransform();
}

void onKeyboard(uint32_t type, uint8_t state, SDL_Keysym keysym) {
	switch (keysym.sym) {
	case SDLK_F1:
		g_text_size = 96;
		break;
	case SDLK_F2:
		g_text_size = 20;
		break;
	case SDLK_F3:
		break;
	case SDLK_F4:
		break;
	case SDLK_F5:
		g_rasterizer->dumpDebugData();
		break;
	case SDLK_F11:
		g_ui.window->setFullScreen(true);
		break;
	default:
		break;
	}
}

void onTextInput(const char *text) {

	printf("TextInput: %s\n", text);

	auto key = text[0];

	if (!g_rasterizer->isTextInputStarted()) {
		if (key == 61) {
			//g_inputController._lastPos = glm::ivec2(x, g_viewHeight - 1 - y);
			//g_inputController.wheel(1.f);
			//updateRasterizerTransform();
			return;
		}
		else if (key == 45) {
			//g_inputController._lastPos = glm::ivec2(x, g_viewHeight - 1 - y);
			//g_inputController.wheel(-1.f);
			//updateRasterizerTransform();
			return;
		}
		else if (key == ',') {
			g_brush_size -= 2.f;
			if (g_brush_size>64.f) {
				g_brush_size = 64.f;
			}
			else if (g_brush_size<4.f) {
				g_brush_size = 4.f;
			}
			return;
		}
		else if (key == '.') {
			g_brush_size += 2.f;
			if (g_brush_size>64.f) {
				g_brush_size = 64.f;
			}
			else if (g_brush_size<4.f) {
				g_brush_size = 4.f;
			}
			return;
		}
		else if (key == '\r') {
			if (g_activatedTool == AT_PEN) {
				g_activatedTool = AT_BRUSH;
			}
			else {
				g_activatedTool = AT_PEN;
			}
			g_inputController.setControllerMode(Mochimazui::CCM_NULL);
			return;
		}
	}
	else {
		//vec3 scenep = glut_scene_rev_proj(
		//	glut_view_rev_proj(vec2(x, y)));
		//auto scenep = g_click_scene_pos;
		//if (key >= ' '&&key < 127 && (g_activatedTool == AT_PEN || g_activatedTool == AT_BRUSH) && !g_rasterizer->isTextInputStarted()) {
		//	g_rasterizer->startTextInput(scenep.x, scenep.y, g_brush_size*1.5f, g_penColor);
		//}
		g_rasterizer->onCharMessage(key);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void setupGLDebug() {

	if (!Config::GLDebug()) { 
		glDisable(GL_DEBUG_OUTPUT);
		return; 
	}

	GLGUI_CHECK_GL_ERROR();
	glEnable(GL_DEBUG_OUTPUT);
	auto cb = [](GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const GLchar *msg,
		const void *data) {

		static const std::map<uint32_t, std::string> s_type = {
			{ GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR,"Deprecated behavior" },
			{ GL_DEBUG_TYPE_ERROR,"Error" },
			{ GL_DEBUG_TYPE_MARKER,"Marker" },
			{ GL_DEBUG_TYPE_OTHER,"Other" },
			{ GL_DEBUG_TYPE_PERFORMANCE,"Performance" },
			{ GL_DEBUG_TYPE_POP_GROUP,"Pop group" },
			{ GL_DEBUG_TYPE_PORTABILITY,"Portablity" },
			{ GL_DEBUG_TYPE_PUSH_GROUP,"Push group" },
			{ GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR,"Undefined behavior" }
		};

		static const std::map<uint32_t, std::string> s_severity = {
			{ GL_DEBUG_SEVERITY_HIGH,"High" },
			{ GL_DEBUG_SEVERITY_LOW, "Low" },
			{ GL_DEBUG_SEVERITY_MEDIUM, "Medium" },
			{ GL_DEBUG_SEVERITY_NOTIFICATION, "Notification" }
		};

		switch (type) {
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		case GL_DEBUG_TYPE_ERROR:
		case GL_DEBUG_TYPE_MARKER:
		case GL_DEBUG_TYPE_OTHER:
		case GL_DEBUG_TYPE_PERFORMANCE:
		case GL_DEBUG_TYPE_POP_GROUP:
		case GL_DEBUG_TYPE_PORTABILITY:
		case GL_DEBUG_TYPE_PUSH_GROUP:
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			break;
		default:
			return;
		}

		printf("-------- -------- -------- --------\nGL Debug\n");
		printf("-> Type: %s ID: %d Severity: %s\n", 
			s_type.find(type)->second.c_str(), id, s_severity.find(severity)->second.c_str());
		printf("%s\n", msg);
		printf("-------- -------- -------- --------\n");
	};

	glDebugMessageCallback(cb, nullptr);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void loadVG() {

	if (Config::Animation()) { return; }

	std::shared_ptr<VGContainer> vgc;

	{
		auto inputFile = Config::InputFile();
		auto suffix = Mochimazui::stdext::string(inputFile).right(3);

		if (suffix == "svg") {
			SVG svg;
			svg.setA128(Config::A128());
			svg.load(inputFile);
			vgc = svg.vgContainer();
		}
		else if (suffix == "rvg") {
			RVG rvg;
			rvg.setA128(Config::A128());
			rvg.load(inputFile);
			vgc = rvg.vgContainer();
		}
		else if (suffix == "vgt" || suffix == "vgb") {
			vgc.reset(new VGContainer);
			vgc->load(inputFile);
		}
		else {
			error_printf("loadVG: illegal input file: \"%s\"", inputFile.c_str());
			printf("\n");
		}
	}
	
	if (Config::MergeAdjacentPath()) {
		vgc->mergeAdjacentPath();
	}

	g_vg_size = vec2(vgc->width, vgc->height);

	//
	g_rasterizer->addVg(*vgc);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void initUI() {

	std::string uiFile = Config::MinimalUI() ? "./ui/minimal_ui.json" : "./ui/ui.json";
	g_ui.window = g_app->windows().createWindowFromJSONFile(uiFile).lock();
	if (!g_ui.window) {
		error_printf("GLGUI error: window not created.\n");
		return;
	}

	g_ui.displayWindow = g_ui.window->ui<SubWindow>("display");
	auto spDisplayWindow = g_ui.displayWindow.lock();
	if (!spDisplayWindow) {
		error_printf("UI error: ui object \"display\" not found.\n");
		return;
	}

	auto pos = spDisplayWindow->position();
	auto size = spDisplayWindow->size();
	g_window_pos = glm::ivec2(pos.x, pos.y);
	g_window_size = glm::ivec2(size.x, size.y);

	spDisplayWindow->onPaint(onPaint);
	spDisplayWindow->onResize(onResize);

	spDisplayWindow->onMouseLeftButtonDown(onLeftButtonDown);
	spDisplayWindow->onMouseLeftButtonUp(onLeftButtonUp);
	spDisplayWindow->onMouseRightButtonDown(onRightButtonDown);
	spDisplayWindow->onMouseRightButtonUp(onRightButtonUp);
	spDisplayWindow->onMouseMove(onMouseMove);
	spDisplayWindow->onMouseWheel(onMouseWheel);
	spDisplayWindow->onTextInput(onTextInput);
	spDisplayWindow->onKeyboard(onKeyboard);
}

void initRasterizer() {

	if (Config::Verbose()) {
		auto pipeline_name = Mochimazui::ras_pipeline_mode_to_string(Config::PipelineMode());
		color_printf(12, 0, "ras: %s\n", pipeline_name.c_str());
	}

	g_rasterizer = Mochimazui::createRasterizer(Config::PipelineMode());

	if (!g_rasterizer) {
		error_printf("initRasterizer: cannot create rasterizer\n");
	}

	// init rasterizer
	g_rasterizer->verbose(Config::Verbose());
	g_rasterizer->drawCurve(Config::DrawCurve());
	g_rasterizer->setMultisampleOutput(Config::MultisampleOutput());
	g_rasterizer->setSamples(Config::Samples());
	g_rasterizer->useMaskTable(Config::UseMaskTable());
	g_rasterizer->enableSRGBCorrection(Config::sRGB());
	g_rasterizer->verticalFlipOutputFile(Config::OutputVerticalFlip());
	g_rasterizer->breakBeforeGL(Config::BreakBeforeGL());
	g_rasterizer->countPixel(Config::CountPixel());
	g_rasterizer->setFPS(Config::ShowFPS());
	g_rasterizer->setAnimation(Config::Animation());

	if (Config::SaveOutputFile()) {
		g_rasterizer->saveOutputToFile(Config::OutputFile());
	}

	if (Config::TigerClip()) {
		g_rasterizer->enableTigerClip(true);
		g_rasterizer->enableStencilBuffer(true);
	}

	g_rasterizer->reserveInk(Config::ReserveInk());

	// !!! set ALL config value before calling init !!!
	g_rasterizer->init();
}

void updateView() {
	if (Config::FitWindowToVGSize()) {
		g_app->windows().window(0)->resize((int)g_vg_size.x, (int)g_vg_size.y);
	}

	auto cfg_window_size = Config::WindowSize();
	auto cfg_output_size = Config::OutputSize();

	if (cfg_window_size.x && cfg_window_size.y) {
		g_window_size = cfg_window_size;
		g_app->windows().window(0)->resize(g_window_size.x, g_window_size.y);
	}

	if (cfg_output_size.x && cfg_output_size.y) {
		g_output_size = cfg_output_size;
	}
	else {
		g_output_size = g_window_size;
	}

	g_rasterizer->viewport(g_window_pos.x, g_window_pos.y,
		g_window_size.x, g_window_size.y);
	g_rasterizer->resizeOutputBuffer(g_output_size.x, g_output_size.y);

	g_inputController.init((int)g_vg_size.x, (int)g_vg_size.y);
	g_outputController.init((int)g_vg_size.x, (int)g_vg_size.y);

	g_inputController.setControllerMode(Mochimazui::CCM_MOVE);
	g_outputController.setControllerMode(Mochimazui::CCM_MOVE);

	if (Config::FitVGToWindowSize()) {
		g_inputController.fitToView(g_output_size.x, g_output_size.y);
	}
	updateRasterizerTransform();

}

//#define USE_NO_GUI_MAIN
#ifdef USE_NO_GUI_MAIN
// -------- -------- -------- -------- -------- -------- -------- --------
int no_glgui_main(int argc, char *argv[]) {

	try {

		// get configuration.
		Config::load("default.cfg");
		Config::parseArg(argc, (const char **)argv);
		if (Config::Help()) { Mochimazui::help(); return 0; }
		if (Config::ListFiles()) { return 0; }

		if (Config::Verbose()) {
			color_printf(0xF, 0, "\n---- vg_rasterizer ----\n\n");
		}

		// init random seed
		srand(clock());

		// run test
		if (Config::Test()) {
			Test::implicit_curve_test(); return 0;
		}

		//

		bool init = !(SDL_Init(SDL_INIT_EVERYTHING) < 0);
		if (!init) {
			printf("SDLApp: %s\n", SDL_GetError());
		}

		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);

		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
		int samples = 32;
		while (samples) {
			auto f = SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, samples);
			if (!f) { printf("SDL_GL_MULTISAMPLESAMPLES: %d\n", samples); break; }
			samples >>= 1;
		}

		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);

		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

		auto sdlWindow = SDL_CreateWindow("NV_command_list",
			SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
			1920, 1080, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

		if (!sdlWindow) {
			printf("Window::createSDLWindow: %s\n", SDL_GetError());
		}                

		auto sdlGLContext = SDL_GL_CreateContext(sdlWindow);
		if (!sdlGLContext) {
			printf("Window::createSDLWindow: %s\n", SDL_GetError());
		}

		if (ogl_LoadFunctions() == ogl_LOAD_FAILED) {
			printf("ogl_LoadFunctions Error.\n");
		}

		SDL_GL_SetSwapInterval(1);

		// 
		setupGLDebug();

		//
		initRasterizer();

		// load file.
		loadVG();

		// update view.
		g_rasterizer->viewport(g_window_pos.x, g_window_pos.y, g_window_size.x, g_window_size.y);

		g_inputController.init(g_vg_size.x, g_vg_size.y);
		//g_inputController.fitToView(g_window_size.x, g_window_size.y);
		g_inputController.setControllerMode(Mochimazui::CCM_MOVE);

		g_outputController.init(g_vg_size.x, g_vg_size.y);
		g_outputController.setControllerMode(Mochimazui::CCM_MOVE);

		updateRasterizerTransform();

		// run
		SDL_Event e;

		bool quit = false;
		while (!quit) {
			while (SDL_PollEvent(&e) != 0) {
			}
			g_rasterizer->rasterize();
			SDL_GL_SwapWindow(sdlWindow);
		}

		return 0;
	}
	catch (std::exception &e) {
		printf("\n");
		Mochimazui::stdext::error_printf("Exception in main:\n%s\n", e.what());
		return 0;
	}
	catch (...) {
		printf("\n");
		Mochimazui::stdext::error_printf("Unknown exception in main\n");
		return 0;
	}
}
#else
// -------- -------- -------- -------- -------- -------- -------- --------
int main(int argc, char *argv[]) {
	try 
	{
		if (Mochimazui::init_config(argc, argv)) { return 0; }

		if (Config::Verbose()) {
			color_printf(0xF, 0, "---- vg_rasterizer ----\n");
			printf("thrust version: %d.%d.%d\n", 
				THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION);
		}

		// init random seed
		srand(clock());

		// init application.
		g_app.reset(new Application(SDL_INIT_EVENTS | SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE | SDL_INIT_TIMER, 
			Config::MultisampleOutput() ? Config::Samples() : 1));
		if (!g_app->initialized()) { return 0; }
		g_app->enableIdleEvent(true);

		// init ui.
		initUI();

		// 
		setupGLDebug();

		//
		initRasterizer();

		//
		loadVG();

		// update view.
		updateView();
		
		if (Config::Benchmark() || Config::CountPixel()) {
			benchmark();
			return 0;
		}

		if (Config::StepTiming()) {
			g_rasterizer->enableStepTiming(true);
			step_timing();
			return 0;
		}

		// run
		g_app->run();

		return 0;
	}
	catch (std::exception &e) {
		printf("\n");
		Mochimazui::stdext::error_printf("Exception in main:\n%s\n", e.what());
		return 1;
	}
	catch (...) {
		printf("\n");
		Mochimazui::stdext::error_printf("Unknown exception in main\n");
		return 2;
	}
}
#endif
