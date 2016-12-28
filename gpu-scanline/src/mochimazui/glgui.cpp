
#define _CRT_SECURE_NO_WARNINGS

#include "glgui.h"

#pragma warning(push)
#pragma warning(disable: 4312)
//#define STB_IMAGE_IMPLEMENTATION
//#define STBI_ONLY_PNG
#include "3rd/stb_image.h"
#pragma warning(pop)

//#define STB_TRUETYPE_IMPLEMENTATION
#include "3rd/stb_truetype.h"

#include <cassert>
#include <ctime>
#include <list>

namespace Mochimazui {
namespace GLGUI {

// -------- -------- -------- -------- -------- -------- -------- --------
UIObject::UIObject() {
	_margin.left = _margin.right = _margin.top = _margin.bottom = 0;
	_border.left = _border.right = _border.top = _border.bottom = 0;
	_padding.left = _padding.right = _padding.top = _padding.bottom = 0;
	_layoutSpace = 2;
}

UIObject::UIObject(const std::weak_ptr<UIObject> &pparent)
	: _obj_parent(pparent) {
	_margin.left = _margin.right = _margin.top = _margin.bottom = 0;
	_border.left = _border.right = _border.top = _border.bottom = 0;
	_padding.left = _padding.right = _padding.top = _padding.bottom = 0;
	_layoutSpace = 2;
}

void UIObject::show(bool f) {
	_show = f;
	auto pp = _obj_parent.lock();
	if (pp) {
		pp->arrangeLayout();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::move(int w, int h) {}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::resize(int w, int h) {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::moveAndResize(int x, int y, int w, int h) {}

// -------- -------- -------- -------- -------- -------- -------- --------
bool UIObject::hitTest(const IVec2 &p) {
	return (_pos.x <= p.x) && (_pos.y <= p.y) &&
		(p.x < (_pos.x + _size.x)) && (p.y < (_pos.y + _size.y));
}

// -------- -------- -------- -------- -------- -------- -------- --------
std::shared_ptr<UIObject> UIObject::findChild(const IVec2 &p) {
	// TODO: improve performance by quad-tree?
	for (auto pc : _obj_children) {
		if (pc->hitTest(p)) { 
			auto pf = pc->findChild(p);
			if (pf) { return pf; }
			return pc;
		}
	}
	return nullptr;
}

// -------- -------- -------- -------- -------- -------- -------- --------
IVec2 UIObject::layoutSize() {
	if (_show) {
		int w = -1;
		int h = -1;
		if (_sizePolicy.x == SP_FIX) {
			w = _size.x + _margin.left + _margin.right;
		}
		if (_sizePolicy.y == SP_FIX) {
			h = _size.y + _margin.top + _margin.bottom;
		}
		return IVec2(w, h);
	} else {
		return IVec2(0, 0);
	}
}

IVec2 UIObject::minSize() {
	int cw = 0;
	int ch = 0;

	if (_obj_id == "ok-cancel") {
		cw = cw;
	}

	for (auto pc : _obj_children) {
		auto ms = pc->minSize();
		cw = std::max(cw, ms.x);
		ch = std::max(ch, ms.y);
	}

	if (_sizePolicy.x == SP_FIX) { cw = std::max(cw, _size.x); }
	if (_sizePolicy.y == SP_FIX) { ch = std::max(ch, _size.y); }

	cw += _border.left + _padding.left + _padding.right + _border.right;
	ch += _border.top + _padding.top + _padding.bottom + _border.bottom;

	return IVec2(cw, ch);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::arrangeLayout() {

	// SizePolicy: fix, auto, min, expand,

	//
	// this-> padding
	// child -> layoutSize
	// [
	//   this->layoutSpace
	//   child2 -> layoutSize
	// ] *
	// this-> padding

	// (layoutSize == -1) -> auto.
	//

	if (_layout == WL_VERTICAL || _layout == WL_HORIZONTAL) {

		std::function<SizePolicy(const SizePolicy2&)> gsp;
		std::function<int(const IVec2&)> gi;
		std::function<int(const ISize2&)> gs;

		if (_layout == WL_HORIZONTAL) {
			gsp = [](const SizePolicy2 &sp) ->SizePolicy {return sp.x; };
			gi = [](const IVec2 &v) -> int {return v.x; };
			gs = [](const ISize2 &v) -> int {return v.w; };
		}
		else {
			gsp = [](const SizePolicy2 &sp) ->SizePolicy {return sp.y; };
			gi = [](const IVec2 &v) -> int {return v.y; };
			gs = [](const ISize2 &v) -> int {return v.h; };
		}

		// ------- -------
		// common part
		auto &cs = _obj_children;

		int autoCount = 0;
		int expandCount = 0;
		int showCount = 0;
		int minCount = 0;
		int fixCount = 0;

		int used = 0;

		std::vector<SizePolicy> csp(cs.size());
		std::vector<int> csize(cs.size());

		// get size policy & count number.
		for (int i = 0; i < cs.size(); ++i) {
			auto pc = cs[i];
			auto sp = gsp(pc->_sizePolicy);
			csp[i] = sp;
			if (pc->_show) {
				++showCount;
				if (sp == SP_FIX) {
					++fixCount;
					csize[i] = gi(pc->size());
				}
				else {
					csize[i] = gi(pc->minSize());
					if (sp == SP_AUTO) { ++autoCount; }
					else if (sp == SP_EXPAND) { ++expandCount; }
					else if (sp == SP_MIN) { ++minCount; }
				}
				used += csize[i];
			}
		}

		//
		if (expandCount) {
			autoCount = 0;
			for (auto &sp : csp) {
				if (sp == SP_AUTO) {
					++minCount;
					sp = SP_MIN;
				}
				else if (sp == SP_EXPAND) {
					++autoCount;
					sp = SP_AUTO;
				}
			}
		}

		//
		int unused = gi(_size)
			- gs(_border.size2() + _padding.size2())
			- used - (showCount - 1) * _layoutSpace;

		for (int i = 0; i < cs.size(); ++i) {
			if (csp[i] == SP_AUTO) {
				auto psize = unused / autoCount;
				csize[i] += psize;
				unused -= psize;
				--autoCount;
			}
		}

		// -------- --------
		if (_layout == WL_HORIZONTAL) {
			int offset = 0;
			offset += _pos.x + _border.left + _padding.left;
			int yy = _pos.y + _border.top + _padding.top;
			int yyy = _size.y - (_border.top + _padding.top + _padding.bottom + _border.bottom);

			for (int i = 0; i < _obj_children.size(); ++i) {
				auto c = _obj_children[i];
				if (!c->_show) { continue; }
				c->_pos.x = offset + c->_margin.left;
				c->_pos.y = yy + c->_margin.top;

				auto new_width = csize[i] - c->_margin.left - c->_margin.right;;
				auto new_height = yyy;
				c->resizeEvent(new_width, new_height);

				offset += csize[i] + _layoutSpace;
			}
		}
		else {
			int offset = 0;
			offset += _pos.y + _border.top + _padding.top;
			int xx = _pos.x + _border.left + _padding.left;
			int xxx = _size.x - (_border.left + _padding.left + _padding.right + _border.right);

			for (int i = 0; i < _obj_children.size(); ++i) {
				auto c = _obj_children[i];
				if (!c->_show) { continue; }

				c->_pos.x = xx + c->_margin.left;
				c->_pos.y = offset + c->_margin.top;

				auto new_width = xxx;
				auto new_height = csize[i] - c->_margin.top - c->_margin.bottom;
				c->resizeEvent(new_width, new_height);

				offset += csize[i] + _layoutSpace;
			}
		}

	}

	for (auto &p : _obj_children) {
		p->arrangeLayout();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::repaint() {
	_repaint = true;
}

void UIObject::repaintEvent() {
	if (_repaint) {
		paintEvent();
		_repaint = false;
		for (auto pc : _obj_children) {
			pc->repaint();
			pc->repaintEvent();
		}
	}
	else {
		for (auto pc : _obj_children) {
			pc->repaintEvent();
		}
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::attributeFromJSON(const nlohmann::json &j) {
	using PRIVATE::dget;

	_obj_id = dget<std::string>(j, "id", "window");

	_pos.x = dget<int>(j, "x", _pos.x);
	_pos.y = dget<int>(j, "y", _pos.y);
	_size.x = dget<int>(j, "width", _size.x);
	_size.y = dget<int>(j, "height", _size.y);

	std::string layout = dget<std::string>(j, "layout", "horizontal");
	if (layout == "horizontal") {
		_layout = WL_HORIZONTAL;
	} else if (layout == "vertical") {
		_layout = WL_VERTICAL;
	} else {
		// ERROR.
	}

	auto getSizePolicy = [&](const std::string &sps, SizePolicy def) {
		if (sps == "auto") { return SP_AUTO; }
		else if (sps == "fix") { return SP_FIX; }
		else if (sps == "min") { return SP_MIN; }
		else if (sps == "expand") { return SP_EXPAND; }
		else { return def; }
	};

	_sizePolicy.x = getSizePolicy(dget<std::string>(j, "size-policy-x"), _sizePolicy.x);
	_sizePolicy.y = getSizePolicy(dget<std::string>(j, "size-policy-y"), _sizePolicy.y);
}

std::weak_ptr<UIObject> UIObject::ui_by_id(const std::string &id) {
	if (!id.length()) { return std::weak_ptr<UIObject>(); }
	if (id[0] == '#') {
		// global name
		auto i = _obj_children_map.find(id);
		if (i == _obj_children_map.end()) { 
			return std::weak_ptr<UIObject>(); 
		}
		return i->second;
	} else {
		// hierarchical name
		auto dotPos = id.find('.');
		if (dotPos == std::string::npos) {
			auto i = _obj_children_map.find(id);
			if (i == _obj_children_map.end()) { return std::weak_ptr<UIObject>(); }
			return i->second;
		} else {
			std::string id0 = id.substr(0, dotPos);
			std::string id1 = id.substr(dotPos + 1);
			auto i = _obj_children_map.find(id0);
			if (i == _obj_children_map.end()) { return std::weak_ptr<UIObject>(); }

			//return i->second->
			return std::weak_ptr<UIObject>();
		}
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void UIObject::paintChildren() {
	for (auto &p : _obj_children) {
		p->paintEvent();
	}
}

void UIObject::idleEvent() {
	if (_onIdle) { _onIdle(); }
}

void UIObject::paintEvent() {
	if (_onPaint) {
		_onPaint();
	}
	paintChildren();
}

void UIObject::resizeEvent(int w, int h) {
	_size.x = w;
	_size.y = h;
	if (_onResize) {
		_onResize(w, h);
	}
	arrangeLayout();
}

void UIObject::mouseEnterEvent() {
	if (_onMouseEnter) { _onMouseEnter(); }
}

void UIObject::mouseLeaveEvent() {
	if (_onMouseLeave) { _onMouseLeave(); }
}

void UIObject::mouseLeftButtonDownEvent(int x, int y) {
	if (_onMouseLeftButtonDown) { _onMouseLeftButtonDown(x, y); }
}

void UIObject::mouseLeftButtonUpEvent(int x, int y) {
	if (_onMouseLeftButtonUp) { _onMouseLeftButtonUp(x, y); }
}

void UIObject::mouseMiddleButtonDownEvent(int x, int y) {
	if (_onMouseMiddleButtonDown) { _onMouseMiddleButtonDown(x, y); }
}

void UIObject::mouseMiddleButtonUpEvent(int x, int y) {
	if (_onMouseMiddleButtonUp) { _onMouseMiddleButtonUp(x, y); }
}

void UIObject::mouseRightButtonDownEvent(int x, int y) {
	if (_onMouseRightButtonDown) { _onMouseRightButtonDown(x, y); }
}

void UIObject::mouseRightButtonUpEvent(int x, int y) {
	if (_onMouseRightButtonUp) { _onMouseRightButtonUp(x, y); }
}

void UIObject::mouseWheelEvent(int x, int y) {
	if (_onMouseWheel) { _onMouseWheel(x, y); }
}

void UIObject::mouseMoveEvent(int x, int y, uint32_t buttonState) {
	if (_onMouseMove) { _onMouseMove(x, y, buttonState); }
}

void UIObject::textInputEvent(const char *text) {
	if (_onTextInput) { _onTextInput(text); }
}

void UIObject::keyboardEvent(uint32_t type, uint8_t state, SDL_Keysym keysym) {
	if (_onKeyboard) {
		_onKeyboard(type, state, keysym);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
Window::Window() {}
Window::Window(const std::shared_ptr<WindowManager> &pmanager)
	:_window_manager(pmanager)
{}
Window::~Window() {
	SDL_GL_DeleteContext(_sdlGLContext);
	SDL_DestroyWindow(_sdlWindow);
}

void Window::resize(int w, int h) {
	if (_sdlWindow) {
		SDL_SetWindowSize(_sdlWindow, w, h);
	}
}

void Window::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_title = dget<std::string>(j, "title", "GLGUI");
}

void Window::arrangeLayout() {
	for (auto psw : _obj_children) {
		psw->resizeEvent(_size.x, _size.y);
	}
}

void Window::resizeEvent(int w, int h) {
	_size = IVec2(w, h);
	for (auto psw : _obj_children) {
		psw->resizeEvent(_size.x, _size.y);
	}
	repaint();
}

void Window::mouseLeftButtonDownEvent(int x, int y) {
	if (_currentChild) {
		auto p = _currentChild->position();
		_currentChild->mouseLeftButtonDownEvent(x - p.x, y - p.y);
	}
}

void Window::mouseLeftButtonUpEvent(int x, int y) {
	if (_currentChild) {
		auto p = _currentChild->position();
		_currentChild->mouseLeftButtonUpEvent(x - p.x, y - p.y);
	}
}

void Window::mouseMoveEvent(int x, int y, uint32_t buttonState) {
	auto c = findChild(IVec2(x, y));
	auto e = [&]() {
		c->mouseEnterEvent();
		auto p = c->position();
		c->mouseMoveEvent(x - p.x, y - p.y, buttonState);
	};
	if (!c) {
		if (_currentChild) { e(); }
	} else if (c == _currentChild) {
		if (_currentChild) { e(); }
	} else {
		if (_currentChild) { _currentChild->mouseLeaveEvent(); }
		e();
		_currentChild = c;
	}
}

void Window::mouseWheelEvent(int x, int y) {
	if (_currentChild) {
		_currentChild->mouseWheelEvent(x, y);
	}
}

void Window::textInputEvent(const char *text) {
	if (_currentChild) {
		_currentChild->textInputEvent(text);
	}
}

void Window::keyboardEvent(uint32_t type, uint8_t state, SDL_Keysym keysym) {
	if (_currentChild) {
		_currentChild->keyboardEvent(type, state, keysym);
	}
}

std::shared_ptr<SharedPaintData> Window::sharedPaintData() {
	if (!_sharedPaintData) {
		_sharedPaintData.reset(new SharedPaintData);
		_sharedPaintData->init();
	}
	return _sharedPaintData;
}

void Window::repaintEvent() {
	paintEvent();
}

std::weak_ptr<UIObject> Window::ui_by_id(const std::string &id) {
	if (_obj_children.size()) {
		return _obj_children[0]->ui_by_id(id);
	}
	else {
		return std::weak_ptr<UIObject>();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
std::shared_ptr<Window> Window::createWindowFromJSON(const std::string &json_str) {
	return createWindowFromJSON(json_str, nullptr);
}

std::shared_ptr<Window> Window::createWindowFromJSONFile(const std::string &fileName) {
	return createWindowFromJSONFile(fileName, nullptr);
}

std::shared_ptr<Window> Window::createWindowFromJSON(const std::string &json_str,
	std::function<void(std::shared_ptr<Window>&pw)> onCreate) {

	using nlohmann::json;
	using PRIVATE::dget;

	json j;
	try {
		j = json::parse(json_str);
	}
	catch (std::exception &e) {
		printf("In createUi: %s\n", e.what());
		return nullptr;
	}

	auto w = j["window"];
	if (w.is_null()) {
		printf("createUi: Warning: \"window\" not found.\n");
		return nullptr;
	}

	std::shared_ptr<Window> pWindow(new Window());
	pWindow->attributeFromJSON(w);
	pWindow->createSDLWindow();

	// create subwindow.
	auto subWindowList = w["subwindows"];

	if (!subWindowList.is_null()) {

		auto subWindowRefList = j["subwindows"];

		// create an empty agency.
		std::shared_ptr<SubWindow> prsw(new SubWindow(pWindow));
		prsw->_obj_id = pWindow->_obj_id;
		prsw->_size = pWindow->_size;
		prsw->_layout = pWindow->_layout;
		prsw->_border = ISize4(0);
		pWindow->_obj_children.push_back(prsw);

		std::function<void(const json &list, std::shared_ptr<SubWindow> &pp)> createSubWindow;
		
		createSubWindow = [&createSubWindow, &subWindowRefList, &pWindow]
			(const json &list, std::shared_ptr<SubWindow> &pp) {

			//
			for (auto &swj : list) {
				std::string type = dget<std::string>(swj, "type", "");
				std::shared_ptr < SubWindow > psw; 

				if (type.empty()) { psw.reset(new SubWindow(pp)); }
				else if (type == "subwindow") { psw.reset(new SubWindow(pp)); }
				else if (type == "scroll-window") { psw.reset(new ScrollWindow(pp)); }
				else if (type == "horizontal-layout" || type == "hlayout") { psw.reset(new HorizontalLayout(pp)); }
				else if (type == "vertical-layout" || type == "vlayout") { psw.reset(new VerticalLayout(pp)); }
				else if (type == "horizontal-spacer") { psw.reset(new HorizontalSpacer(pp)); }
				else if (type == "vertical-spacer") { psw.reset(new VerticalSpacer(pp)); }
				else if (type == "frame") { psw.reset(new Frame(pp)); }
				else if (type == "horizontal-line") { psw.reset(new HorizontalLine(pp)); }
				else if (type == "vertical-line") { psw.reset(new VerticalLine(pp)); }
				else if (type == "label") { psw.reset(new Label(pp)); }
				else if (type == "push-button") { psw.reset(new PushButton(pp)); }
				else if (type == "radio-button") { psw.reset(new RadioButton(pp)); }
				else if (type == "check-box") { psw.reset(new CheckBox(pp)); }
				else if (type == "horizontal-slider") { psw.reset(new HorizontalSlider(pp)); }
				else {
					printf("In createWindowFromJSON:\n\tError: illegal type\"%s\".\n", type.c_str());
					psw.reset(new SubWindow(pp));
				}
				psw->attributeFromJSON(swj);

				//
				// id: ascii string without '.'
				//   start with '#' : global name.
				//   else: local name.
				//     
				auto &id = psw->id();
				if (id.empty()) {
					printf("In createWindowFromJSON: subwindow has empty id.\n");
					continue;
				}
				if (id[0] == '#') {
					// global name
					auto &cmap = pWindow->_obj_children_map;
					auto i = cmap.find(id);
					if (i != cmap.end()) {
						printf("In createWindowFromJSON: id \"%s\" already used.\n", id.c_str());
						continue;
					}
					cmap[id] = psw;
				}
				else {
					// local name
					auto &cmap = pp->_obj_children_map;
					auto i = cmap.find(id);
					if (i != cmap.end()) {
						printf("In createWindowFromJSON: id \"%s\" already used.\n", id.c_str());
						continue;
					}
					cmap[id] = psw;
				}
				pp->_obj_children.push_back(psw);

				auto subWindowList = swj["subwindows"];
				if (!subWindowList.is_null()) {
					createSubWindow(subWindowList, psw);
				}

			}
		};
		createSubWindow(subWindowList, prsw);
	}
	if (onCreate) {
		onCreate(pWindow);
	}
	pWindow->arrangeLayout();
	pWindow->repaint();
	return pWindow;
}

std::shared_ptr<Window> Window::createWindowFromJSONFile(const std::string &fileName,
	std::function<void(std::shared_ptr<Window>&pw)> onCreate) {

	std::shared_ptr<Window> pWindow;
	char *json_text = nullptr;

	FILE *fin;
	fin = fopen(fileName.c_str(), "rb");
	if (!fin) {
		printf("createWindowFromJSONFile: cannot open input file.");
		return nullptr;
	}

	fseek(fin, 0, SEEK_END);
	auto size = ftell(fin);
	json_text = new char[size + 1];
	if (!json_text) {
		printf("createWindowFromJSONFile: cannot open input file.");
		return nullptr;
	}
	fseek(fin, 0, SEEK_SET);
	size_t size_read = fread(json_text, 1, size, fin);
	json_text[size] = '\0';

	try {
		pWindow = Window::createWindowFromJSON(json_text, onCreate);
	}
	catch (std::exception &e) {
		printf("In createUiFromFile: %s\n", e.what());
	}

	delete[] json_text;

	return pWindow;
}

void Window::idleEvent() {
	UIObject::idleEvent();
}

void Window::paintEvent() {

	GLGUI_CHECK_GL_ERROR();

	// count fps & set to window title.
	//static std::list<int> timestamps;
	//int now = clock();
	//while (!timestamps.empty() && timestamps.front() + 1000 < now) {
	//	timestamps.pop_front();
	//}
	//timestamps.push_back(now);

	//float fps = 1000.0f / ((timestamps.back() - timestamps.front()) / (float)timestamps.size());
	//char fpss[128];
	//sprintf(fpss, "Frame per second: %.2f, Time per frame: %.2f", fps, 1000.f / std::max(1.f, fps));
	//auto newTitle = _title + " VG: " + fpss;
	//SDL_SetWindowTitle(_sdlWindow, newTitle.c_str());

	// paint.

	//glDisable(GL_SCISSOR_TEST);
	//glViewport(0, 0, _size.x, _size.y);
	//glClearColor(.25f, .25f, .25f, 1.f);
	//glClear(GL_COLOR_BUFFER_BIT);

	UIObject::paintEvent();
	SDL_GL_SwapWindow(_sdlWindow);

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void Window::createSDLWindow() {

	_sdlWindow = SDL_CreateWindow(_title.c_str(), 
		//SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, 32,
		_size.x, _size.y, 
		//SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
		SDL_WINDOW_OPENGL);

	if (!_sdlWindow) {
		printf("Window::createSDLWindow: %s\n", SDL_GetError());
	}

	_sdlGLContext = SDL_GL_CreateContext(_sdlWindow);
	if (!_sdlGLContext) {
		printf("Window::createSDLWindow: %s\n", SDL_GetError());
	}

	if (ogl_LoadFunctions() == ogl_LOAD_FAILED) {
		printf("ogl_LoadFunctions Error.\n");
	}
	//GLGUI_CHECK_GL_ERROR();
	//auto e = glewInit();
	//GLGUI_CHECK_GL_ERROR();
	//if (e != GLEW_OK) {
	//	printf("GLEW ERROR.");
	//}

	//SDL_SetWindowFullscreen(_sdlWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);

	SDL_GL_SetSwapInterval(1);
}

// -------- -------- -------- -------- -------- -------- -------- --------
bool Window::eventHandle(const SDL_Event &e) {

	switch (e.type) {

	case SDL_WINDOWEVENT:
		switch (e.window.event) {
		case SDL_WINDOWEVENT_RESIZED:
		case SDL_WINDOWEVENT_SIZE_CHANGED:
			resizeEvent(e.window.data1, e.window.data2);
			return true;
		default:
			break;
		}
		break;

	case SDL_MOUSEBUTTONDOWN:
		if (!_currentChild) { break; }
		switch (e.button.button)
		{
		case SDL_BUTTON_LEFT:
			mouseLeftButtonDownEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_MIDDLE:
			_currentChild->mouseMiddleButtonDownEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_RIGHT:
			_currentChild->mouseRightButtonDownEvent(e.button.x, e.button.y);
			return true;
		default:
			break;
		}
		break;
	case SDL_MOUSEBUTTONUP:
		if (!_currentChild) { break; }
		switch (e.button.button)
		{
		case SDL_BUTTON_LEFT:
			mouseLeftButtonUpEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_MIDDLE:
			_currentChild->mouseMiddleButtonUpEvent(e.button.x, e.button.y);
			return true;
		case SDL_BUTTON_RIGHT:
			_currentChild->mouseRightButtonUpEvent(e.button.x, e.button.y);
			return true;
		default:
			break;
		}
		break;
	case SDL_MOUSEMOTION:
		mouseMoveEvent(e.motion.x, e.motion.y, e.motion.state);
		break;
	case SDL_MOUSEWHEEL:
		mouseWheelEvent(e.wheel.x, e.wheel.y);
		return true;
		break;
	case SDL_KEYDOWN:
		keyboardEvent(SDL_KEYDOWN, e.key.state, e.key.keysym);
		break;
	case SDL_TEXTINPUT:
		textInputEvent(e.text.text);
		return true;
		break;
	case SDL_QUIT:
		// ignore.
		break;
	}
	return false;
}

// -------- -------- -------- -------- -------- -------- -------- --------

SubWindow::SubWindow(const std::weak_ptr<Window> &pparent)
	:UIObject(pparent), _root_window(pparent)
{}

SubWindow::SubWindow(const std::weak_ptr<SubWindow> &pparent)
	:UIObject(pparent), _root_window(pparent.lock()->_root_window)
{}

std::weak_ptr<PaintData> SubWindow::paintData() {
	if (!_paintData) { _paintData.reset(new PaintData); }
	return _paintData;
}

//void SubWindow::paintEvent() {
//	paintChildren();
//}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Window manager

// -------- -------- -------- -------- -------- -------- -------- --------
void WindowManager::addWindow(std::shared_ptr<Window> &pw) {
	if (pw) { _windows.push_back(pw); }
}

// -------- -------- -------- -------- -------- -------- -------- --------
std::weak_ptr<Window> WindowManager::createWindow() {
	return createWindowFromJSON("{}");
}

std::weak_ptr<Window> WindowManager::createWindowFromJSON(const std::string &json_str) {
	auto pw = Window::createWindowFromJSON(json_str);
	addWindow(pw);
	return pw;
}

std::weak_ptr<Window> WindowManager::createWindowFromJSONFile(const std::string &fileName) {
	auto pw = Window::createWindowFromJSONFile(fileName);
	addWindow(pw);
	return pw;
}

void WindowManager::repaintEvent() {
	for (auto pw : _windows) {
		pw->repaintEvent();
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void WindowManager::idleEvent() {
	for (auto &pw : _windows) {
		pw->idleEvent();
	}
}

bool WindowManager::eventHandle(const SDL_Event &e) {
	// TODO: dispatch event by window id.
	if (_windows.size()) {
		return _windows[0]->eventHandle(e);
	}
	return false;
}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Application
long long g_tick0, g_freq;
void zeroShitTime() {
	QueryPerformanceCounter((LARGE_INTEGER*)&g_tick0);
	QueryPerformanceFrequency((LARGE_INTEGER*)&g_freq);
}
double getShitTime() {
	long long g_tick1;
	QueryPerformanceCounter((LARGE_INTEGER*)&g_tick1);
	return double(g_tick1 - g_tick0) / double(g_freq);
}
//#define PTIME() printf("%.2lf: %s %d\n",getShitTime()*1000.0,__FILE__,__LINE__)
#define PTIME() 

void Application::run() {

	SDL_Event e;
	auto handle = [&]() {
		if (!_wm.eventHandle(e) && e.type == SDL_QUIT) {
			_quit = true;
			return false;
		}
		return true;
	};

	while (!_quit) {
		zeroShitTime();
		if (_idle) {
			while (SDL_PollEvent(&e) != 0) {
				if (!handle()) { break; }
			}

			PTIME();
			//if (SDL_PollEvent(&e) != 0) {
			//	PTIME();
			//	if (!handle()) { break; }
			//	PTIME();
			//}
			PTIME();
			_wm.idleEvent();
			PTIME();
			_wm.repaintEvent();
			PTIME();
		}
		else {
			int flag = SDL_WaitEvent(&e);
			if (flag) {
				handle();
			}
			else {
				printf("Application::run: SDL_WaitEvent error.\n");
			}
			while (SDL_PollEvent(&e) != 0) {
				if (!handle()) { break; }
			}			
			_wm.repaintEvent();
		}		
		
	}
}

void Application::init(Uint32 flags, int samples) {
	_init = !(SDL_Init(flags) < 0);
	if (!_init) {
		printf("SDLApp: %s\n", SDL_GetError());
	}

	auto ret = 0;

	// TODO:
	// move to window manager ?
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);

	auto flag = SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, samples);
	if (flag) { 
		char emsg[128];
		sprintf(emsg, "SDL_GL_SetAttribute: cannot set SDL_GL_MULTISAMPLESAMPLES to %d\n", samples);
		throw std::runtime_error(emsg);
	}

	//SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	/* Turn on double buffering with a 24bit Z buffer.
	* You may need to change this to 16 or 32 for your system */	
	ret = SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	ret = SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	// ??? ogl init failed after adding this line. ???
	//ret = SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalLayout::HorizontalLayout(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent) {
	_sizePolicy = SizePolicy2(SP_AUTO, SP_AUTO);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

VerticalLayout::VerticalLayout(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent)
{
	_sizePolicy = SizePolicy2(SP_AUTO, SP_AUTO);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalSpacer::HorizontalSpacer(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent) 
{
	_sizePolicy = SizePolicy2(SP_EXPAND, SP_FIX);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

VerticalSpacer::VerticalSpacer(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent)
{
	_sizePolicy = SizePolicy2(SP_FIX, SP_EXPAND);
	_margin = _border = _padding = ISize4(0);
	_size = IVec2(0, 0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalScrollBar::HorizontalScrollBar(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{}

// -------- -------- -------- -------- -------- -------- -------- --------
VerticalScrollBar::VerticalScrollBar(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent)
{}

// -------- -------- -------- -------- -------- -------- -------- --------
ScrollWindow::ScrollWindow(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{}

// -------- -------- -------- -------- -------- -------- -------- --------
Frame::Frame(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{}

void Frame::paintEvent() {
	Painter painter(this);
	painter.strokeWidth((float)_borderWidth);
	painter.strokeColor(_borderColor);
	painter.fillColor(_backgroundColor);
	painter.rectangle(0.f, 0.f, (float)_size.x, (float)_size.y);
	paintChildren();
}

void Frame::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);	
	using PRIVATE::dget;
	_borderWidth = dget<int>(j, "border-width", 1);
	_border.left = _border.right = _border.top = _border.bottom = _borderWidth;
}

// -------- -------- -------- -------- -------- -------- -------- --------
// UIHorizontalLine
HorizontalLine::HorizontalLine(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{
	_sizePolicy.y = SP_FIX;
	_size.y = _width;
}
void HorizontalLine::paintEvent() {
	Painter painter(this);
	painter.strokeWidth((float)_width);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	float y = _width * 0.5f;
	painter.line(0.f, y, (float)_size.x, y);
}
void HorizontalLine::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_width = dget<int>(j, "line-width", _width);
	_size.y = _width;
}

// -------- -------- -------- -------- -------- -------- -------- --------
// UIVerticalLine
VerticalLine::VerticalLine(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent)
{
	_sizePolicy.x = SP_FIX;
	_size.x = _width;
}
void VerticalLine::paintEvent() {
	Painter painter(this);
	painter.strokeWidth((float)_width);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	float x = _width * 0.5f;
	painter.line(x, 0.f, x, (float)_size.y);
}
void VerticalLine::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_width = dget<int>(j, "line-width", _width);
	_size.x = _width;
}

// -------- -------- -------- -------- -------- -------- -------- --------
Label::Label(const std::weak_ptr<SubWindow> &pparent) :SubWindow(pparent)
{
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void Label::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
}

void Label::paintEvent() {
	Painter painter(this);

	//painter.strokeColor(RGBA(255, 255, 255, 255));
	//painter.fillColor(RGBA(63, 63, 63, 255));
	//painter.rectangle(0, 0, _size.x, _size.y);

	IRect rect;
	rect.x = _border.left + _padding.left;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

IVec2 Label::minSize() {
	return IVec2(1, 20);
}

// -------- -------- -------- -------- -------- -------- -------- --------
PushButton::PushButton(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent){
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void PushButton::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
}

void PushButton::paintEvent() {

	Painter painter(this);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	painter.strokeWidth(1);

	if (_mouseIn) {
		painter.fillColor(RGBA(127, 127, 127, 255));
	}
	else {
		painter.fillColor(RGBA(63, 63, 63, 255));
	}
	painter.rectangle(0.f, 0.f, (float)_size.x, (float)_size.y);

	//painter.rectangle(1, 1, _size.x - 2, _size.y - 2);

	IRect rect;
	rect.x = _border.left + _padding.left;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

void PushButton::mouseLeftButtonDownEvent(int x, int y) {
	if (_onClick) { _onClick(); }
}

void PushButton::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void PushButton::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

IVec2 PushButton::minSize() {
	auto ms = _border.size2() + _padding.size2();
	Painter painter(this);
	auto tsize = painter.textSize(_text);
	//return IVec2(ms.w + tsize.x, ms.h + tsize.y);
	return IVec2(ms.w + tsize.x, 20);
}

void PushButton::onClick(std::function<void(void)> cb) {
	_onClick = cb;
}

// -------- -------- -------- -------- -------- -------- -------- --------
RadioButton::RadioButton(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent) {
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void RadioButton::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
	_value = dget<std::string>(j, "value") == "true" ? true : false;
}

void RadioButton::paintEvent() {

	Painter painter(this);

	//
	float cx = 10;
	float cy = 10;

	if (_mouseIn) {
		painter.strokeColor(0);
		painter.fillColor(RGBA(127, 127, 127, 255));
		painter.circle(cx, cy, 6.f);
	}

	painter.strokeColor(RGBA(255, 255, 255, 255));
	painter.fillColor(0);
	painter.circle(cx, cy, 6.f);

	if (_value) {
		painter.strokeColor(0);
		painter.fillColor(RGBA(255, 255, 255, 255));
		painter.circle(cx, cy, 3.f);
	}

	int width = 10 * (int)_text.size();

	IRect rect;
	rect.x = _border.left + _padding.left + 16;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

void RadioButton::mouseLeftButtonDownEvent(int x, int y) {
	_value = !_value;
	if (_onClick) { _onClick(_value); }
}

void RadioButton::onClick(std::function<void(bool)> cb) {
	_onClick = cb;
}

void RadioButton::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void RadioButton::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

// -------- -------- -------- -------- -------- -------- -------- --------
CheckBox::CheckBox(const std::weak_ptr<SubWindow> &pparent)
	: SubWindow(pparent) {
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void CheckBox::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_text = dget<std::string>(j, "text");
	_value = dget<std::string>(j, "value") == "true" ? true : false;
}

void CheckBox::paintEvent() {
	Painter painter(this);
	
	float cx = 10;
	float cy = 10;
	float S = 6.f;

	painter.strokeColor(RGBA(255, 255, 255, 255));
	if (_mouseIn) {
		painter.fillColor(RGBA(127, 127, 127, 255));
	}
	else {
		painter.fillColor(RGBA(63, 63, 63, 255));
	}
	painter.rectangle(cx - S, cy - S, S * 2, S * 2);

	S = 3.f;
	if (_value) {
		painter.fillColor(RGBA(255, 255, 255, 255));
		painter.rectangle(cx - S, cy - S, S * 2, S * 2);
	}

	int width = 10 * (int)_text.size();

	IRect rect;
	rect.x = _border.left + _padding.left + 16;
	rect.y = 0;
	rect.width = _size.x;
	rect.height = _size.y;
	painter.text(rect, _text, HA_LEFT);
}

void CheckBox::mouseLeftButtonDownEvent(int x, int y) {
	_value = !_value;
	if (_onClick) { _onClick(_value); }
}

void CheckBox::onClick(std::function<void(bool)> cb) {
	_onClick = cb;
}

void CheckBox::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void CheckBox::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

// -------- -------- -------- -------- -------- -------- -------- --------
HorizontalSlider::HorizontalSlider(const std::weak_ptr<SubWindow> &pparent)
	:SubWindow(pparent) {
	_sizePolicy.y = SP_FIX;
	_size.y = 20;
}

void HorizontalSlider::attributeFromJSON(const nlohmann::json &j) {
	UIObject::attributeFromJSON(j);
	using PRIVATE::dget;
	_value = dget<int>(j, "value", _value);
	_maxValue = dget<int>(j, "max-value", _maxValue);
}

void HorizontalSlider::paintEvent() {
	Painter painter(this);
	if (_mouseIn) {
		painter.strokeWidth(5);
		painter.strokeColor(RGBA(127, 127, 127, 255));
		painter.line(10.f, 10.f, _size.x - 10.f, 10.f);
	}
	painter.strokeWidth(1);
	painter.strokeColor(RGBA(255, 255, 255, 255));
	painter.line(10.f, 10.f, _size.x - 10.f, 10.f);
	painter.strokeColor(RGBA(255, 255, 255, 0));
	painter.fillColor(RGBA(255, 255, 255, 255));
	painter.circle(10 + (_size.x - 20) * (float)_value / (float)_maxValue, 10, 6);
}

void HorizontalSlider::mouseEnterEvent() {
	_mouseIn = true;
	repaint();
}

void HorizontalSlider::mouseLeaveEvent() {
	_mouseIn = false;
	repaint();
}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Painter Base

// -------- -------- -------- -------- -------- -------- -------- --------
PainterBase::PainterBase(SubWindow *psw) {
	if (!psw) { return; }
	init(psw);
}

// -------- -------- -------- -------- -------- -------- -------- --------
PainterBase::PainterBase(std::weak_ptr<SubWindow> &wpsw) {
	auto psw = wpsw.lock();
	if (!psw) { return; }
	init(psw.get());
}

// -------- -------- -------- -------- -------- -------- -------- --------
PainterBase::~PainterBase() {
	update();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void PainterBase::init(SubWindow *psw) {

	GLGUI_CHECK_GL_ERROR();

	// Get paint data.
	_data = psw->paintData().lock();
	_sharedData = psw->rootWindow().lock()->sharedPaintData();

	// Set viewport.
	const IVec2 &pos = psw->position();
	const IVec2 &size = psw->size();
	_size = size;
	IVec2 rootSize;
	auto prw = psw->rootWindow().lock();
	if (!prw) {
		printf("PainterBase::init: SubWindow doesn't have a root.\n");
	}
	rootSize = prw ? prw->size() : size;

	int x = pos.x;
	int y = rootSize.y - (pos.y + size.y);
	int w = std::max(size.x, 0);
	int h = std::max(size.y, 0);

	//
	glEnable(GL_SCISSOR_TEST);
	glDisable(GL_DEPTH_TEST);
	glScissor(x, y, w, h);
	glViewport(x, y, w, h);
	if (w == 0 || h == 0) { return; }

	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
	//glOrtho(0, w, 0, h, -1, 1);

	//glTranslatef(0, h, 0);
	//glScalef(0, -1, 0);

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void PainterBase::update() {
	if (!_update) { return; }
	_update = false;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void PainterBase::strokeColor(const RGBA &c) {
	_strokeColor = c;
}
void PainterBase::strokeColor(uint8_t a) {
	_strokeColor.a = a;
}
void PainterBase::strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_strokeColor = RGBA(r,g,b,a);
}

void PainterBase::strokeWidth(float w) {
	_strokeWidth = w;
}

void PainterBase::fillColor(const RGBA &c) {
	_fillColor = c;
}
void PainterBase::fillColor(uint8_t a) {
	_fillColor.a = a;
}
void PainterBase::fillColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_fillColor = RGBA(r, g, b, a);
}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// GLES painter.
#ifdef GLGUI_USE_GLES_PAINTER

void GLESSharedPaintData::init() {

	// font
	if (!fontBuffer) {
#ifdef ENABLE_STB_TEXT
		auto &ttf_buffer = fontBuffer;
		if (!ttf_buffer) { ttf_buffer = new uint8_t[50000]; }
		//auto &font = font;
		fread(ttf_buffer, 1, 50000, fopen("./font/DroidSans.ttf", "rb"));
		//fread(ttf_buffer, 1, 50000, fopen("./font/DroidSans-Bold.ttf", "rb"));
		stbtt_InitFont(&font, ttf_buffer, stbtt_GetFontOffsetForIndex(ttf_buffer, 0));
#endif
	}

	// shader
	std::string uiVertexShader =
		"#version 400 \n"
		"uniform float width; \n"
		"uniform float height; \n"
		"layout(location = 0) in vec2 i_vertex; \n"
		"layout(location = 1) in vec4 i_color; \n"
		"flat out vec4 fragColor; \n"		
		"void main(){ \n"
		"    fragColor = i_color; \n"
		"    gl_Position = vec4(i_vertex.x/width*2.0-1.0, i_vertex.y/height*2.0-1.0, 0.0, 1.0);"
		"}";

	std::string uiFragmentShader =
		"#version 400 \n"
		"layout(location = 0) out vec4 color; \n"
		"flat in vec4 fragColor; \n"
		"void main(){ \n"
		"    color = fragColor;"
		"}";

	std::string textVertexShader =
		"#version 400 \n"
		"uniform float width; \n"
		"uniform float height; \n"
		"layout(location = 0) in vec2 i_vertex; \n"
		"layout(location = 1) in vec2 i_texcoord; \n"
		"out vec2 texcoord; \n"		
		"void main(){ \n"
		"    texcoord = i_texcoord; \n"
		"    gl_Position = vec4(i_vertex.x/width*2.0-1.0, i_vertex.y/height*2.0-1.0, 0.0, 1.0);"
		"}";

	std::string textFragmentShader =
		"#version 400 \n"
		"uniform sampler2D ftex; \n"
		"in vec2 texcoord; \n"
		"layout(location = 0) out vec4 color; \n"
		"void main(){ \n"
		"    color = texture(ftex, texcoord);"
		"}";

	uiShader
		.createShader(GLPP::Vertex, uiVertexShader)
		.createShader(GLPP::Fragment, uiFragmentShader)
		.link();

	textShader
		.createShader(GLPP::Vertex, textVertexShader)
		.createShader(GLPP::Fragment, textFragmentShader)
		.link();

	_ready = true;
}

// -------- -------- -------- -------- -------- -------- -------- --------
GLESPainter::GLESPainter(SubWindow *psw)
	:PainterBase(psw) {

	auto &uiShader = _sharedData->uiShader;
	if (!uiShader.linked()) {uiShader.link(); }
	uiShader.uniform1f("width", (float)_size.x);
	uiShader.uniform1f("height", (float)_size.y);

	auto &textShader = _sharedData->textShader;
	if (!textShader.linked()) { textShader.link(); }
	textShader.uniform1f("width", (float)_size.x);
	textShader.uniform1f("height", (float)_size.y);
	textShader.uniform1i("ftex", 0);

	glUseProgram(0);

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
GLESPainter::GLESPainter(std::weak_ptr<SubWindow> &wpsw)
	:PainterBase(wpsw){
}

// -------- -------- -------- -------- -------- -------- -------- --------
GLESPainter::~GLESPainter() {
	update();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::update() {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::line(float x0, float y0, float x1, float y1) {
	
	GLGUI_CHECK_GL_ERROR();

	if (_strokeColor.a) {

		float vd[4] = {
			x0, y0, x1, y1
		};
		float cd[8] = {
			_strokeColor.r / 255.f, _strokeColor.g / 255.f, _strokeColor.b / 255.f, _strokeColor.a / 255.f,
			_strokeColor.r / 255.f, _strokeColor.g / 255.f, _strokeColor.b / 255.f, _strokeColor.a / 255.f,
		};

		glLineWidth(_strokeWidth);

		_sharedData->uiShader.use();

		GLuint va;
		glGenVertexArrays(1, &va);
		glBindVertexArray(va);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		GLuint vb, cb;
		glGenBuffers(1, &vb);
		glGenBuffers(1, &cb);
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glVertexAttribPointer(1, 4, GL_FLOAT, 0, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float), vd, GL_STATIC_DRAW);
		
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), cd, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDrawArrays(GL_LINES, 0, 2);
		glFinish();

		glBindVertexArray(0);

		glDeleteBuffers(1, &vb);
		glDeleteBuffers(1, &cb);
		glDeleteVertexArrays(1, &va);

		glUseProgram(0);
	}

	GLGUI_CHECK_GL_ERROR();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::circle(float cx, float cy, float r) {

	_sharedData->uiShader.use();

	GLuint va;
	glGenVertexArrays(1, &va);
	glBindVertexArray(va);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	GLuint vb, cb;
	glGenBuffers(1, &vb);
	glGenBuffers(1, &cb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, cb);
	glVertexAttribPointer(1, 4, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	static const double PI = acos(-1.0);
	glLineWidth(1);

	if (_strokeColor.a) {
		std::vector<float> vd;
		std::vector<float> cd;

		for (int i = 0; i <= 64; ++i) {
			vd.push_back((float)(cx + sin(i / 32.0 *PI) * r));
			vd.push_back((float)(cy + cos(i / 32.0 *PI) * r));
			cd.push_back(_strokeColor.r/255.f);
			cd.push_back(_strokeColor.g / 255.f);
			cd.push_back(_strokeColor.b / 255.f);
			cd.push_back(_strokeColor.a / 255.f);
		}
		int32_t s = (int32_t)vd.size();
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), vd.data(), GL_STATIC_DRAW);
		s = (int32_t)cd.size();
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), cd.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDrawArrays(GL_LINE_STRIP, 0, s / 2);
		glFinish();
	}

	if (_fillColor.a) {
		std::vector<float> vd;
		std::vector<float> cd;
		vd.push_back(cx);
		vd.push_back(cy);
		cd.push_back(_fillColor.r/ 255.f);
		cd.push_back(_fillColor.g / 255.f);
		cd.push_back(_fillColor.b / 255.f);
		cd.push_back(_fillColor.a / 255.f);
		for (int i = 0; i <= 64; ++i) {
			vd.push_back((float)(cx + sin(i / 32.0 *PI) * r));
			vd.push_back((float)(cy + cos(i / 32.0 *PI) * r));
			cd.push_back(_fillColor.r / 255.f);
			cd.push_back(_fillColor.g / 255.f);
			cd.push_back(_fillColor.b / 255.f);
			cd.push_back(_fillColor.a / 255.f);
		}
		int32_t s = (int32_t)vd.size();
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), vd.data(), GL_STATIC_DRAW);
		s = (int32_t)cd.size();
		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), cd.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDrawArrays(GL_TRIANGLE_FAN, 0, s / 2);
		glFinish();
	}

	glBindVertexArray(0);

	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &cb);
	glDeleteVertexArrays(1, &va);

	glUseProgram(0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::rectangle(float x, float y, float w, float h) {

	_sharedData->uiShader.use();

	GLuint va;
	glGenVertexArrays(1, &va);
	glBindVertexArray(va);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	GLuint vb, cb;
	glGenBuffers(1, &vb);
	glGenBuffers(1, &cb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, cb);
	glVertexAttribPointer(1, 4, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (_strokeColor.a && (_strokeWidth>0)) {

		//glColor4ub(_strokeColor.r, _strokeColor.g, _strokeColor.b, _strokeColor.a);

		float vd[8] = {
			x, y,
			x + w, y,
			x + w, y + h,
			x, y + h
		};

		std::vector<float> cd;
		cd.push_back(_strokeColor.r / 255.f);
		cd.push_back(_strokeColor.g / 255.f);
		cd.push_back(_strokeColor.b / 255.f);
		cd.push_back(_strokeColor.a / 255.f);
		cd.insert(cd.end(), cd.begin(), cd.end());
		cd.insert(cd.end(), cd.begin(), cd.end());
		cd.insert(cd.end(), cd.begin(), cd.end());
		
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), vd, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, cb);
		glBufferData(GL_ARRAY_BUFFER, 32 * sizeof(float), cd.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_QUADS, 0, 4);
		glFinish();

		auto &s = _strokeWidth;
		x += s;
		y += s;
		w -= s * 2;
		h -= s * 2;
	}

	float vd[8] = {
		x, y,
		x + w, y,
		x + w, y + h,
		x, y + h
	};

	std::vector<float> cd;
	cd.push_back(_fillColor.r / 255.f);
	cd.push_back(_fillColor.g / 255.f);
	cd.push_back(_fillColor.b / 255.f);
	cd.push_back(_fillColor.a / 255.f);
	cd.insert(cd.end(), cd.begin(), cd.end());
	cd.insert(cd.end(), cd.begin(), cd.end());
	cd.insert(cd.end(), cd.begin(), cd.end());

	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), vd, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, cb);
	glBufferData(GL_ARRAY_BUFFER, 32 * sizeof(float), cd.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_QUADS, 0, 4);
	glFinish();

	//
	glBindVertexArray(0);

	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &cb);
	glDeleteVertexArrays(1, &va);

	glUseProgram(0);
}

// -------- -------- -------- -------- -------- -------- -------- --------
#ifdef ENABLE_STB_TEXT
stbtt_fontinfo &GLESPainter::getSharedFont() {
	auto ps = _sharedData;
	if (!ps->ready()) {
		ps->init();
	}
	return ps->font;
}

// -------- -------- -------- -------- -------- -------- -------- --------
IVec2 GLESPainter::textSize(const stbtt_fontinfo &font, const std::string &t) {
#ifdef ENABLE_STB_TEXT
	int ascent, descent, advance, lsb;
	float xoffset = 0.f, scale = stbtt_ScaleForPixelHeight(&font, 16);
	stbtt_GetFontVMetrics(&font, &ascent, &descent, 0);
	for (int ch = 0; t[ch]; ++ch) {
		float x_shift = xoffset - (float)floor(xoffset);
		stbtt_GetCodepointHMetrics(&font, t[ch], &advance, &lsb);
		stbtt_GetCodepointBitmapBoxSubpixel(&font, t[ch], scale, scale, x_shift, 0, 0, 0, 0, 0);
		xoffset += (advance * scale);
		if (t[ch + 1]) {
			xoffset += scale*stbtt_GetCodepointKernAdvance(&font, t[ch], t[ch + 1]);
		}
	}
	int SH = (int)((std::abs(ascent) + std::abs(descent)) * scale);
	int SW = (int)(xoffset + 1);
	return IVec2(SW, SH);
#else
	return IVec2(0, 0);
#endif
}

IVec2 GLESPainter::textSize(const std::string &t) {
	auto &font = getSharedFont();
	return textSize(font, t);
}

#else
IVec2 GLESPainter::textSize(const std::string &t) {
	return IVec2(0, 0);
}
#endif

// -------- -------- -------- -------- -------- -------- -------- --------
void GLESPainter::text(float x, float y, const std::string &t) {
#ifdef ENABLE_STB_TEXT
	GLGUI_CHECK_GL_ERROR();

	// 
	auto &font = getSharedFont();
	auto scale = stbtt_ScaleForPixelHeight(&font, 16);
	int ascent, descent, lineGap;
	stbtt_GetFontVMetrics(&font, &ascent, &descent, &lineGap);
	auto baseline = (int)(ascent*scale);

	//
	float xoffset = 0.f;
	int ch = 0;
	for (int ch = 0; t[ch]; ++ch) {
		int advance, lsb;
		float x_shift = xoffset - (float)floor(xoffset);
		stbtt_GetCodepointHMetrics(&font, t[ch], &advance, &lsb);
		stbtt_GetCodepointBitmapBoxSubpixel(&font, t[ch], scale, scale, x_shift, 0, 0, 0, 0, 0);
		xoffset += (advance * scale);
		if (t[ch + 1]) {
			xoffset += scale*stbtt_GetCodepointKernAdvance(&font, t[ch], t[ch + 1]);
		}
	}
	int SH = (int)((std::abs(ascent) + std::abs(descent)) * scale);
	int SW = (int)(xoffset + 1);

	uint8_t *screen = new uint8_t[SW*SH]; 
	memset(screen, 0, SW*SH);

	float xpos = 0;
	ch = 0;
	while (t[ch]) {
		int advance, lsb, x0, y0, x1, y1;
		float x_shift = xpos - (float)floor(xpos);
		stbtt_GetCodepointHMetrics(&font, t[ch], &advance, &lsb);
		stbtt_GetCodepointBitmapBoxSubpixel(&font, t[ch], scale, scale, x_shift, 0, 
			&x0, &y0, &x1, &y1);
		stbtt_MakeCodepointBitmapSubpixel(&font, &screen[(baseline + y0) * SW + (int)xpos + x0], 
			x1 - x0, y1 - y0, SW, scale, scale, x_shift, 0, t[ch]);
		// note that this stomps the old data, so where character boxes overlap (e.g. 'lj') it's wrong
		// because this API is really for baking character bitmaps into textures. if you want to render
		// a sequence of characters, you really need to render each bitmap to a temp buffer, then
		// "alpha blend" that into the working buffer
		xpos += (advance * scale);
		if (t[ch + 1])
			xpos += scale*stbtt_GetCodepointKernAdvance(&font, t[ch], t[ch + 1]);
		++ch;
	}

	uint8_t *rgba = new uint8_t[SW*SH * 4];
	for (int j = 0; j < SH; ++j) {
		for (int i = 0; i < SW; ++i) {
			auto p = j*SW + i;
			auto p4 = p * 4;
			auto v = screen[p];
			if (v) {
				rgba[p4] = rgba[p4 + 1] = rgba[p4 + 2] = 255;
				rgba[p4 + 3] = v;
			}
			else {
				rgba[p4] = rgba[p4 + 1] = rgba[p4 + 2] = rgba[p4 + 3] = 0;
			}
		}
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_TEXTURE_2D);

	GLuint tex = 0;
	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SW, SH, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

	_sharedData->textShader.use();

	GLuint va;
	glGenVertexArrays(1, &va);
	glBindVertexArray(va);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	GLuint vb, tb;
	glGenBuffers(1, &vb);
	glGenBuffers(1, &tb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, tb);
	glVertexAttribPointer(1, 2, GL_FLOAT, 0, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	float vd[8] = {
		x, y + SH,
		x + SW, y + SH,
		x + SW, y,
		x, y
	};

	float td[8] = {
		0.f, 0.f, 
		1.f, 0.f, 
		1.f, 1.f,
		0.f, 1.f
	};



	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), vd, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, tb);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), td, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_QUADS, 0, 4);
	glFinish();

	//glBegin(GL_QUADS);
	//glTexCoord2f(0.f, 0.f);
	//glVertex2f(x, y+SH);
	//
	//glTexCoord2f(1.f, 0.f);
	//glVertex2f(x+SW, y+SH);
	//
	//glTexCoord2f(1.f, 1.f);
	//glVertex2f(x+SW, y);

	//glTexCoord2f(0.f, 1.f);
	//glVertex2f(x, y);

	//glEnd();
	//glFinish();

	glBindTexture(GL_TEXTURE_2D, 0);

	glDeleteTextures(1, &tex);

	delete[] screen;
	delete[] rgba;

	//
	glBindVertexArray(0);

	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &tb);
	glDeleteVertexArrays(1, &va);

	glUseProgram(0);

	GLGUI_CHECK_GL_ERROR();
#endif
}

void GLESPainter::text(const IRect &rect, const std::string &t) {
	text(rect, t, HA_CENTER, VA_CENTER);
}
void GLESPainter::text(const IRect &rect, const std::string &t, VerticalAlign va) {
	text(rect, t, HA_CENTER, va);
}
void GLESPainter::text(const IRect &rect, const std::string &t, HorizontalAlign ha) {
	text(rect, t, ha, VA_CENTER);
}

void GLESPainter::text(const IRect &rect, const std::string &t,
	HorizontalAlign ha, VerticalAlign va) {

	static const int W = 10;
	static const int H = 16;

	// default font: w 10 h 16
	float x, y;

	if (ha == HA_LEFT) {
		x = (float)rect.x;
	}
	else if (ha == HA_RIGHT) {
		x = (float)(rect.x + (rect.width - (int)t.size() * W));
	}
	else { // center
		x = rect.x + (rect.width - (int)t.size() * W) / 2.f;
	}

	if (va == VA_TOP) {
		y = (float)(rect.y + (rect.height - H));
	}
	else if (va == VA_BOTTOM) {
		y = (float)(rect.y + rect.height);
	}
	else { // center
		y = rect.y + (rect.height - H) / 2.f;
	}

	text(x, y, t);
}

#endif

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// OpenGL painter.
#ifdef GLGUI_USE_GL_PAINTER
#endif

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// NVIDIA_path_rendering painter.
#ifdef GLGUI_USE_NVPR_PAINTER

// -------- -------- -------- -------- -------- -------- -------- --------
NVPRPainter::NVPRPainter(SubWindow *psw) {
	if (!psw) { return; }
	_data = psw->paintData().lock();
}

// -------- -------- -------- -------- -------- -------- -------- --------
NVPRPainter::NVPRPainter(std::weak_ptr<SubWindow> &wpsw) {
	auto psw = wpsw.lock();
	if (!psw) { return; }
	_data = psw->paintData().lock();
}

// -------- -------- -------- -------- -------- -------- -------- --------
NVPRPainter::~NVPRPainter() {
	update();
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::clear() {
	//glDeletePathsNV()
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::update() {
	if (!_update) { return; }
	_update = false;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::strokeColor(uint8_t a){
	_sa = a;
}
void NVPRPainter::fillColor(uint8_t a){
	_fa = a;
}
void NVPRPainter::strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_sr = r; _sg = g; _sb = b; _sa = a;
}
void NVPRPainter::fillColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	_fr = r; _fg = g; _fb = b; _fa = a;
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::addPath(GLsizei ncmd, GLubyte *cmd, GLsizei ncoord, GLfloat *coord) {
	//GLuint path = glGenPathsNV(1);
	//if (!path) { 
	//	return;
	//}
	//glPathCommandsNV(path, ncmd, cmd, ncoord, GL_FLOAT, coord);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::line() {
	//static const GLubyte pathCommands[10] =
	//{ GL_MOVE_TO_NV, GL_LINE_TO_NV, GL_LINE_TO_NV, GL_LINE_TO_NV,
	//GL_LINE_TO_NV, GL_CLOSE_PATH_NV,
	//'M', 'C', 'C', 'Z' };  // character aliases
	//static const GLshort pathCoords[12][2] =
	//{ { 100, 180 }, { 40, 10 }, { 190, 120 }, { 10, 120 }, { 160, 10 },
	//{ 300, 300 }, { 100, 400 }, { 100, 200 }, { 300, 100 },
	//{ 500, 200 }, { 500, 400 }, { 300, 300 } };
	//glPathCommandsNV(pathObj, 10, pathCommands, 24, GL_SHORT, pathCoords);

	/* Before rendering, configure the path object with desirable path
	parameters for stroking.  Specify a wider 6.5-unit stroke and
	the round join style: */

	//glPathParameteriNV(pathObj, GL_PATH_JOIN_STYLE_NV, GL_ROUND_NV);
	//glPathParameterfNV(pathObj, GL_PATH_STROKE_WIDTH_NV, 6.5);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::rectangle(int x, int y, int w, int h) {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void NVPRPainter::paint(std::shared_ptr<NVPRPaintData> &pd) {

	if (!pd) { return; }
	auto &data = *pd;

	int pathNumber = (int)data.glPath.size();
	for (int i = 0; i < pathNumber; ++i) {

		//glClearStencil(0);
		//glStencilMask(~0);
		//glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		//glMatrixLoadIdentityEXT(GL_PROJECTION);
		//glMatrixLoadIdentityEXT(GL_MODELVIEW);
		//glMatrixOrthoEXT(GL_MODELVIEW, 0, 500, 0, 400, -1, 1);

		GLuint pathObj = data.glPath[i];
		auto fillColor = data.pathFillColor[i];
		auto strokeColor = data.pathStrokeColor[i];

		bool filling = !!(fillColor >> 24);
		bool stroking = !!(strokeColor >> 24);

		if (filling) {

			/* Stencil the path: */

			//glStencilFillPathNV(pathObj, GL_COUNT_UP_NV, 0x1F);

			/* The 0x1F mask means the counting uses modulo-32 arithmetic. In
			principle the star's path is simple enough (having a maximum winding
			number of 2) that modulo-4 arithmetic would be sufficient so the mask
			could be 0x3.  Or a mask of all 1's (~0) could be used to count with
			all available stencil bits.

			Now that the coverage of the star and the heart have been rasterized
			into the stencil buffer, cover the path with a non-zero fill style
			(indicated by the GL_NOTEQUAL stencil function with a zero reference
			value): */

			//glEnable(GL_STENCIL_TEST);
			//glStencilFunc(GL_NOTEQUAL, 0, 0x1F);

			//glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
			//glColor3f(0, 1, 0); // green
			//glCoverFillPathNV(pathObj, GL_BOUNDING_BOX_NV);

		}

		/* The result is a yellow star (with a filled center) to the left of
		a yellow heart.

		The GL_ZERO stencil operation ensures that any covered samples
		(meaning those with non-zero stencil values) are zero'ed when
		the path cover is rasterized. This allows subsequent paths to be
		rendered without clearing the stencil buffer again.

		A similar two-step rendering process can draw a white outline
		over the star and heart. */

		/* Now stencil the path's stroked coverage into the stencil buffer,
		setting the stencil to 0x1 for all stencil samples within the
		transformed path. */

		if (stroking) {

			//glStencilStrokePathNV(pathObj, 0x1, ~0);

			/* Cover the path's stroked coverage (with a hull this time instead
			of a bounding box; the choice doesn't really matter here) while
			stencil testing that writes white to the color buffer and again
			zero the stencil buffer. */

			//glColor3f(1, 1, 0); // yellow
			//glCoverStrokePathNV(pathObj, GL_CONVEX_HULL_NV);

			/* In this example, constant color shading is used but the application
			can specify their own arbitrary shading and/or blending operations,
			whether with Cg compiled to fragment program assembly, GLSL, or
			fixed-function fragment processing.

			More complex path rendering is possible such as clipping one path to
			another arbitrary path.  This is because stencil testing (as well
			as depth testing, depth bound test, clip planes, and scissoring)
			can restrict path stenciling. */
		}
	}

	int textNumber = (int)data.text.size();
	for (int i = 0; i < textNumber; ++i) {

		/* STEP 1: stencil message into stencil buffer.  Results in samples
		within the message's glyphs to have a non-zero stencil value. */

		//glDisable(GL_STENCIL_TEST);
		//glStencilFillPathInstancedNV(numUTF8chars,
		//	GL_UTF8_NV, koreanName, glyphBase,
		//	GL_PATH_FILL_MODE_NV, ~0,  // Use all stencil bits
		//	GL_TRANSLATE_X_NV, xoffset);

		/* STEP 2: cover region of the message; color covered samples (those
		with a non-zero stencil value) and set their stencil back to zero. */

		//glEnable(GL_STENCIL_TEST);
		//glStencilFunc(GL_NOTEQUAL, 0, ~0);
		//glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

		//glColor3ub(192, 192, 192);  // gray
		//glCoverFillPathInstancedNV(numUTF8chars,
		//	GL_UTF8_NV, koreanName, glyphBase,
		//	GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
		//	GL_TRANSLATE_X_NV, xoffset);
	}

}

#endif

} // end of namespace GLGUI
} // end of namespace Mochimazui
