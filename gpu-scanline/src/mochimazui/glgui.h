
// ******** ******** ******** ******** ******** ******** ******** ********
// ******** ******** ******** ******** ******** ******** ******** ********
// GLGUI 
// 
// Mochimazui (Rui Li)
// ruili@rui-li.net
// 
// ******** ******** ******** ******** ******** ******** ******** ********
// ******** ******** ******** ******** ******** ******** ******** ********

#pragma once

#ifndef _MOCHIMAZUI_GLGUI_H_
#define _MOCHIMAZUI_GLGUI_H_

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <map>
#include <unordered_map>

#include "3rd/json.hpp"
#include "glpp.h"
#include <GL/GLU.h>

//#define GLGUI_USE_GLES_PAINTER
//#define GLGUI_USE_GL_PAINTER
//#define GLGUI_USE_NVPR_PAINTER

#if !( defined(GLGUI_USE_GLES_PAINTER) \
	|| defined(GLGUI_UES_GL_PAINTER) \
	|| defined(GLGUI_NVPR_PAINTER))
#define GLGUI_USE_GLES_PAINTER
#endif

#ifdef GLGUI_USE_GLES_PAINTER
//#include <SDL_opengles2.h>
//#define GLEW_STATIC
//#include <GL/glew.h>
#include "3rd/gl_4_5_compatibility.h"
#include "3rd/stb_truetype.h"
#endif

#ifdef GLGUI_USE_GL_PAINTER
#include <SDL_opengl.h>
#endif

#ifdef GLGUI_USE_NVPR_PAINTER
#define GLEW_STATIC
#include <GL/glew.h>
#endif

#include <SDL.h>

namespace Mochimazui {

namespace GLGUI {

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------

// GLGUI.

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// All classes.

class Object;
class UIObject;

class WindowManager;
class Window;
class SubWindow;

/*
class VerticalLayout;
class HorizontalLayout;

class VerticalSpacer;
class HorizontalSpacer;

class UIGroupBox;
class UILabel;
class UILineEditor;

class UISlider;

class UILine;

class UIPushButton;
class UIToolButton;
class UICheckBox;

class UIProgressBar;
*/

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Simple types

enum WindowLayout {
	WL_VERTICAL,
	WL_HORIZONTAL,
};

enum Direction {
	Vertical,
	Horizontal
};

enum SizePolicy {
	SP_AUTO,
	SP_MIN,
	SP_FIX,
	SP_EXPAND,
};

struct SizePolicy2 {
	SizePolicy2() {}
	SizePolicy2(SizePolicy ix, SizePolicy iy) :x(ix), y(iy) {}
	SizePolicy x = SP_AUTO;
	SizePolicy y = SP_AUTO;
};

enum HorizontalAlign {
	HA_LEFT,
	HA_CENTER,
	HA_RIGHT
};

enum VerticalAlign {
	VA_TOP,
	VA_CENTER,
	VA_BOTTOM
};

struct Align2 {
	HorizontalAlign h;
	VerticalAlign v;
};

struct Int2 {
	Int2():x(0), y(0){}
	Int2(int32_t v): x(v), y(v) {}
	Int2(int32_t ix, int32_t iy) :x(ix), y(iy) {}
	union {
		struct {
			int32_t x;
			int32_t y;
		};
		struct {
			int32_t w;
			int32_t h;
		};
	};
};

inline Int2 operator + (const Int2 &a, const Int2 &b) {
	return Int2(a.w + b.w, a.h + b.h);
}

typedef Int2 IVec2;
typedef Int2 ISize2;

//struct Float2 {
//	Float2() :x(0), y(0) {}
//	Float2(float v) : x(v), y(v) {}
//	Float2(float ix, float iy) :x(ix), y(iy) {}
//	union {
//		struct {
//			float x;
//			float y;
//		};
//		struct {
//			float w;
//			float h;
//		};
//	};
//};
//
//inline Float2 operator + (const Float2 &a, const Float2 &b) {
//	return Float2(a.w + b.w, a.h + b.h);
//}
//
//struct FVec2 {
//	FVec2() :x(0), y(0) {}
//	FVec2(float _x, float _y) :x(_x), y(_y) {}
//	float x;
//	float y;
//};

struct ISize4 {
	ISize4() :left(0), right(0), top(0), bottom(0) {}
	ISize4(int i) :left(i), right(i), top(i), bottom(i) {}
	ISize4(int lr, int tb) :left(lr), right(lr), top(tb), bottom(tb) {}
	ISize4(int l, int r, int t, int b)
		: left(l), right(r), top(t), bottom(b) {}
	ISize2 size2() { return ISize2(left + right, top + bottom); }
	int left, right, top, bottom;
};

struct IRect {
	int x; int y;
	int width; int height;
};

struct RGBA {
	RGBA() :r(0), g(0), b(0), a(255) {}
	RGBA(uint8_t ir, uint8_t ig, uint8_t ib, uint8_t ia) 
		:r(ir), g(ig), b(ib), a(ia) {}
	uint8_t r, g, b, a;
};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Callback function type.

typedef std::function<void(void)> VoidFunction;

typedef std::function<void(void)> WindowIdleFunction;
typedef std::function<void(void)> WindowPaintFunction;
typedef std::function<void(int w, int h)> WindowResizeFunction;

typedef std::function<void(int key)> WindowKeyFunction;
typedef std::function<void(int id)> WindowTimerFunction;

typedef std::function<void(int x, int y)> WindowMouseFunction;
typedef std::function<void(int x, int y)> WindowMouseWheelFunction;
typedef std::function<void(int x, int y, uint32_t buttonState)> WindowMouseMoveFunction;

typedef std::function<void(const char *text)> WindowTextInputFunction;
typedef std::function<void(uint32_t type, uint8_t state, SDL_Keysym keysym)> WindowKeyboardFunction;

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// private helper function.

namespace PRIVATE {

template <class V2, class V>
inline V getx(const V2 &v2) { return v2.x; }

template <class V2, class V>
inline V gety(const V2 &v2) { return v2.y; }

template<class VT>
inline VT dget(const nlohmann::json &j, const std::string &k) {
	auto v = j[k];
	return v.is_null() ? VT() : v.get<VT>();
}

template<class VT>
inline VT dget(const nlohmann::json &j, const std::string &k, const VT &dv) {
	auto v = j[k];
	return v.is_null() ? dv : v.get<VT>();
}

inline uint32_t make_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	return (((uint32_t)r) << 24)
		| (((uint32_t)g) << 16)
		| (((uint32_t)b) << 8)
		| a;
}

//inline uint32_t make_argb(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
//	return (((uint32_t)a) << 24)
//		| (((uint32_t)r) << 16)
//		| (((uint32_t)g) << 8)
//		| b;
//}

//inline void checkGLError() {
//	glFinish();
//	auto e = glGetError();
//	if (e != GL_NO_ERROR) {
//		auto estr = gluErrorString(e);
//		printf("%s\n", estr);
//		throw std::runtime_error((char*)estr);
//	}
//}

#ifdef _DEBUG
#define GLGUI_CHECK_GL_ERROR() DEBUG_CHECK_GL_ERROR()
#else
#define GLGUI_CHECK_GL_ERROR() ((void)0)
#endif

}

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Object.
// Reserved.

class Object{
};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// UIObject.
// Base class of all UI elements.

class UIObject : public Object {
	friend Window;
public:
	UIObject();
	UIObject(const std::weak_ptr<UIObject> &pparent);

	UIObject(const UIObject &o) = delete;
	UIObject(UIObject &&o) = delete;

	UIObject &operator=(const UIObject &o) = delete;
	UIObject &operator=(UIObject &&o) = delete;

public:
	std::string id() { return _obj_id; }
	void appendChild(const std::shared_ptr<UIObject> &child) { _obj_children.push_back(child); }
	void appendChild(std::shared_ptr<UIObject> &&child) { _obj_children.push_back(child); }

public:
	template <class T>
	std::weak_ptr<T> ui(const std::string &id) {
		return std::dynamic_pointer_cast<T>(ui_by_id(id).lock());
	}

public:
	void show(bool f = true);
	void hide(bool f = true) { show(!f); }

	void enable(bool f = true) { _enable = f; }
	void disable(bool f = true) { enable(!f); }

	virtual void setFullScreen(bool f) { _fullScreen = f; }
	bool isFullScreen() { return _fullScreen; }

	virtual void repaint();

	IVec2 position() { return _pos; }
	ISize2 size() { return _size; }

	virtual void move(int x, int y);
	virtual void resize(int w, int h);
	virtual void moveAndResize(int x, int y, int w, int h);

	const ISize4 &margin() const { return _margin; }
	const ISize4 &border() const { return _border; }
	const ISize4 &padding() const { return _padding; }

public:
	void onIdle(const WindowIdleFunction &pf) { _onIdle = pf; }
	void onPaint(const WindowPaintFunction &pf) { _onPaint = pf; }

	void onResize(const WindowResizeFunction &pf) { _onResize = pf; }

	void onMouseEnter(const VoidFunction &pf) { _onMouseEnter = pf; }
	void onMouseLeave(const VoidFunction &pf) { _onMouseLeave = pf; }

	void onMouseLeftButtonDown(const WindowMouseFunction &pf) { _onMouseLeftButtonDown = pf; }
	void onMouseLeftButtonUp(const WindowMouseFunction &pf) { _onMouseLeftButtonUp = pf; }

	void onMouseMiddleButtonDown(const WindowMouseFunction &pf) { _onMouseMiddleButtonDown = pf; }
	void onMouseMiddleButtonUp(const WindowMouseFunction &pf) { _onMouseMiddleButtonUp = pf; }

	void onMouseRightButtonDown(const WindowMouseFunction &pf) { _onMouseRightButtonDown = pf; }
	void onMouseRightButtonUp(const WindowMouseFunction &pf) { _onMouseRightButtonUp = pf; }

	void onMouseMove(const WindowMouseMoveFunction &pf) { _onMouseMove = pf; }

	void onMouseWheel(const WindowMouseWheelFunction &pf) {	_onMouseWheel = pf;}

	void onTextInput(const WindowTextInputFunction &pf) { _onTextInput = pf; }
	void onKeyboard(const WindowKeyboardFunction &pf) { _onKeyboard = pf; }

protected:

	// GLGUI event
	virtual void idleEvent();
	virtual void paintEvent();
	virtual void repaintEvent();

	// SDL window event
	//virtual void showEvent();
	//virtual void hideEvent();
	//virtual void exposeEvent();
	//virtual void moveEvent(int );
	virtual void resizeEvent(int w, int h);

	virtual void mouseEnterEvent();
	virtual void mouseLeaveEvent();

	virtual void mouseLeftButtonDownEvent(int x, int y);
	virtual void mouseLeftButtonUpEvent(int x, int y);

	virtual void mouseMiddleButtonDownEvent(int x, int y);
	virtual void mouseMiddleButtonUpEvent(int x, int y);

	virtual void mouseRightButtonDownEvent(int x, int y);
	virtual void mouseRightButtonUpEvent(int x, int y);

	virtual void mouseWheelEvent(int x, int y);

	//virtual void mouseWheelDown();
	//virtual void mouseWheelUp();

	//virtual void mouseLeftButtonDown();
	//virtual void mouseLeftButtonUp();
	virtual void mouseMoveEvent(int x, int y ,uint32_t buttonState);

	//virtual void keyPress();
	//virtual void keyRelease();

	virtual void textInputEvent(const char *text);
	virtual void keyboardEvent(uint32_t type, uint8_t state, SDL_Keysym keysym);

protected:
	std::shared_ptr<UIObject> findChild(const IVec2 &p);
	void paintChildren();

	virtual IVec2 layoutSize();
	virtual IVec2 minSize();

	virtual void arrangeLayout();

	virtual bool hitTest(const IVec2 &v);
	virtual void attributeFromJSON(const nlohmann::json &j);
	virtual std::weak_ptr<UIObject> ui_by_id(const std::string &id);

protected:
	std::string _obj_id;
	std::weak_ptr<UIObject> _obj_parent;
	std::vector < std::shared_ptr<UIObject> > _obj_children;
	std::map < std::string, std::weak_ptr<SubWindow> > _obj_children_map;

protected:
	bool _repaint = true;
	bool _show = true;
	bool _enable = true;

	WindowLayout _layout = WL_HORIZONTAL;
	SizePolicy2 _sizePolicy;

	IVec2 _pos = IVec2(0,0);
	IVec2 _size; // margin > size > border > padding

	ISize2 _minSize = ISize2(0,0);
	ISize2 _maxSize;

	ISize4 _margin;
	ISize4 _border;
	ISize4 _padding;
	int _layoutSpace; // space between subwindows.

	bool _fullScreen = false;

protected:
	WindowIdleFunction _onIdle;
	WindowPaintFunction _onPaint;
	WindowResizeFunction _onResize;

	VoidFunction _onMouseEnter;
	VoidFunction _onMouseLeave;

	WindowMouseFunction _onMouseLeftButtonDown;
	WindowMouseFunction _onMouseLeftButtonUp;

	WindowMouseFunction _onMouseMiddleButtonDown;
	WindowMouseFunction _onMouseMiddleButtonUp;

	WindowMouseFunction _onMouseRightButtonDown;
	WindowMouseFunction _onMouseRightButtonUp;

	WindowMouseWheelFunction _onMouseWheel;

	WindowMouseMoveFunction _onMouseMove;

	WindowTextInputFunction _onTextInput;
	WindowKeyboardFunction _onKeyboard;

	// ... more

};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Painters

// -------- -------- -------- -------- -------- -------- -------- --------
// Painter data.

#ifdef GLGUI_USE_GLES_PAINTER

struct GLESSharedPaintData {

	~GLESSharedPaintData() { delete[] fontBuffer; }
	void init();
	bool ready() { return _ready; }

	uint8_t *fontBuffer = 0;
#ifdef ENABLE_STB_TEXT
	stbtt_fontinfo font;
#endif

	GLPP::ShaderProgram uiShader;
	GLPP::ShaderProgram textShader;

private:
	bool _ready = false;
};

struct GLESPaintData {
	// triangles & colors.
	std::vector<float> vertex;
	std::vector<float> color;
};

typedef GLESPaintData PaintData;
typedef GLESSharedPaintData SharedPaintData;

#endif

#ifdef GLGUI_USE_NVPR_PAINTER

// NVPRSharedPaintData.
// Paint data shared by all sub-windows.
struct NVPRSharedPaintData {
	std::unordered_map<std::string, GLuint> fontGlyphBase;
};

// NVPRPaintData.
// Paint data of sub-window.
struct NVPRPaintData {
	std::vector<GLuint> glPath;
	std::vector<uint32_t> pathFillColor;
	std::vector<uint32_t> pathStrokeColor;

	std::vector<std::string> text;
	std::vector<GLuint> textGlyphBase;
	std::vector<float> textSize;
	std::vector<IVec2> textPos;
};

#endif

// -------- -------- -------- -------- -------- -------- -------- --------
// Painter base
class PainterBase {

public:
	PainterBase(std::weak_ptr<SubWindow> &);
	PainterBase(SubWindow *);
	~PainterBase();

public:
	void update();
	void clear();

public:
	void strokeColor(const RGBA &c);
	void strokeColor(uint8_t a);
	void strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

	void strokeWidth(float);

	void fillColor(const RGBA &c);
	void fillColor(uint8_t a);	
	void fillColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

	void textSize(int ts) { _textSize = ts; }

private:
	void init(SubWindow*);

protected:
	RGBA _strokeColor = RGBA(255,255,255,255);
	RGBA _fillColor = RGBA(63, 63, 63, 255);
	float _strokeWidth = 1.f;
	int _textSize;
	bool _update = false;
	std::shared_ptr<PaintData> _data;
	std::shared_ptr<SharedPaintData> _sharedData;

	IVec2 _size;
};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// OpenGL ES painter.
// ...

#ifdef GLGUI_USE_GLES_PAINTER

class GLESPainter : public PainterBase {

public:
	GLESPainter(std::weak_ptr<SubWindow> &);
	GLESPainter(SubWindow *);
	~GLESPainter();

public:
	void update();
	void clear();

public:
	void setFont(const std::string &font);
	
public:
	void line(float x0, float y0, float x1, float y1);
	//void quadLine();
	//void cubicLine();

	//void ellipse();
	void circle(float cx, float cy, float r);
	void rectangle(float x, float y, float w, float h);
	//void roundRectangle();

	IVec2 textSize(const std::string &text);

	void text(float x, float y, const std::string &text);
	void text(const IRect &rect, const std::string &text);
	void text(const IRect &rect, const std::string &text, VerticalAlign va);
	void text(const IRect &rect, const std::string &text, HorizontalAlign ha);
	void text(const IRect &rect, const std::string &text,
		HorizontalAlign ha, VerticalAlign va);

public:
	static void paint(std::shared_ptr<GLESPaintData>&) {};

private:
#ifdef ENABLE_STB_TEXT
	stbtt_fontinfo &getSharedFont();
	IVec2 textSize(const stbtt_fontinfo &font, const std::string &text);
#endif
};

typedef GLESPainter Painter;

#endif

// -------- -------- -------- -------- -------- -------- -------- --------
// OpenGL painter.
// ...
#ifdef GLGUI_USE_GL_PAINTER
#endif

// -------- -------- -------- -------- -------- -------- -------- --------
// NVIDIA_path_rendering painter.
// ...
#ifdef GLGUI_USE_NVPR_PAIN
// -------- -------- -------- -------- -------- -------- -------- --------
// NVPRPainter.
// 
class NVPRPainter {

public:
	NVPRPainter(std::weak_ptr<SubWindow> &);
	NVPRPainter(SubWindow *);
	~NVPRPainter();

public:
	void update();
	void clear();

public:
	void strokeColor(uint8_t a);
	void fillColor(uint8_t a);

	void strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
	void fillColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

public:
	void line();
	void lineTo();
	void quadLine();
	void quadLineTo();
	void cubicLine();
	void cubicLineTo();

	void ellipse();
	void circle();
	void rectangle(int x, int y, int w, int h);
	void roundRectangle();

	void text();

public:
	static void paint(std::shared_ptr<NVPRPaintData> &pd);

private:
	void addPath(GLsizei ncmd, GLubyte *cmd, GLsizei ncoord, GLfloat *coord);

private:
	// stroke color RGBA
	uint8_t _sr = 255, _sg = 255, _sb = 255, _sa = 255;

	// fill color RGBA
	uint8_t _fr = 127, _fg = 127, _fb = 127, _fa = 127;

private:
	bool _update = false;
	std::shared_ptr<NVPRPaintData> _data;
};

typedef NVPRPainter Painter;
typedef NVPRPaintData PaintData;
typedef NVPRSharedPaintData SharedPaintData;

#endif

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Window 
// 
class Window : public UIObject {

	friend WindowManager;
	typedef std::shared_ptr<Window> SharedPtr;

public:
	Window();
	Window(const std::shared_ptr<WindowManager> &pmanager);
	~Window();

public:
	virtual void setFullScreen(bool f) { 
		_fullScreen = f; 
		if (f) {
			SDL_SetWindowFullscreen(_sdlWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);
		}
		else {
		}
	}
	virtual void resize(int w, int h);

public:
	static SharedPtr createWindowFromJSON(const std::string &json_str);
	static SharedPtr createWindowFromJSONFile(const std::string &fileName);
	static SharedPtr createWindowFromJSON(const std::string &json_str,
		std::function<void(std::shared_ptr<Window>&pw)> onCreate);
	static SharedPtr createWindowFromJSONFile(const std::string &fileName,
		std::function<void(std::shared_ptr<Window>&pw)> onCreate);

	//void replaceSubWindow(const std::string &id, const std::shared_ptr<SubWindow>& psw,
	//	bool keepOldChild = false);

	void arrangeLayout();
	void idleEvent();
	void paintEvent();
	void resizeEvent(int w, int h);

	std::shared_ptr<SharedPaintData> sharedPaintData();

	void attributeFromJSON(const nlohmann::json &j);

public:
	void mouseLeftButtonDownEvent(int x, int y);
	void mouseLeftButtonUpEvent(int x, int y);
	void mouseMoveEvent(int x, int y, uint32_t buttonState);
	void mouseWheelEvent(int x, int y);
	void textInputEvent(const char *text);
	void keyboardEvent(uint32_t type, uint8_t state, SDL_Keysym keysym);

protected:
	void repaintEvent();
	virtual std::weak_ptr<UIObject> ui_by_id(const std::string &id);

private:
	void createSDLWindow();
	bool eventHandle(const SDL_Event &e);

private:
	std::string _title;

	std::weak_ptr<WindowManager> _window_manager;
	std::shared_ptr<SharedPaintData> _sharedPaintData;
	std::shared_ptr<UIObject> _currentChild;

	SDL_Window *_sdlWindow = 0;
	SDL_GLContext _sdlGLContext = 0;
};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// SubWindow
//
class SubWindow : public UIObject {
	friend Window;
	friend PainterBase;	
public:
	SubWindow(const std::weak_ptr<Window> &pparent);
	SubWindow(const std::weak_ptr<SubWindow> &pparent);
	//void paintEvent();
	std::weak_ptr<Window> rootWindow() { return _root_window; }
protected:
	std::weak_ptr<PaintData> paintData();

	RGBA _backgroundColor = RGBA(0, 0, 0, 0);
	std::weak_ptr<Window> _root_window;
	std::shared_ptr<PaintData> _paintData;
};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// WindowManamger.
//
class WindowManager : public Object {

public:
	void addWindow(std::shared_ptr<Window> &pw);

	std::weak_ptr<Window> createWindow();
	std::weak_ptr<Window> createWindowFromJSON(const std::string &json_str);
	std::weak_ptr<Window> createWindowFromJSONFile(const std::string &fileName);

	std::shared_ptr<Window> window(int index) {
		return index < _windows.size() ? _windows[index] : nullptr;
	}

	void idleEvent();
	void repaintEvent();
	bool eventHandle(const SDL_Event &e);

protected:
	std::vector<std::shared_ptr<Window>> _windows;
};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Application

class Application : public Object {

public:
	Application() { init(SDL_INIT_EVERYTHING, 32); }
	Application(Uint32 flags) { init(flags, 32); }
	Application(Uint32 flags, int samples) { init(flags, samples); }
	~Application() { SDL_Quit(); }

	bool initialized() { return _init; }
	WindowManager &windows() { return _wm; }
	void enableIdleEvent(bool flag) { _idle = flag; }
	void disableIdleEvent(bool flag) { _idle = !flag; }

	void run();

private:
	void init(Uint32 flags, int samples);

	bool _init = false;
	bool _quit = false;
	bool _idle = false;

	WindowManager _wm;
};

// ******** ******** ******** ******** ******** ******** ******** ********
// -------- -------- -------- -------- -------- -------- -------- --------
// Layout
// 

// -------- -------- -------- -------- -------- -------- -------- --------
//
class HorizontalLayout : public SubWindow {
public:
	HorizontalLayout(const std::weak_ptr<SubWindow> &pparent);
};

// -------- -------- -------- -------- -------- -------- -------- --------
//
class VerticalLayout : public SubWindow {
public:
	VerticalLayout(const std::weak_ptr<SubWindow> &pparent);
};

// -------- -------- -------- -------- -------- -------- -------- --------
class HorizontalSpacer : public SubWindow {
public:
	HorizontalSpacer(const std::weak_ptr<SubWindow> &pparent);
};

class VerticalSpacer : public SubWindow {
public:
	VerticalSpacer(const std::weak_ptr<SubWindow> &pparent);
};

// -------- -------- -------- -------- -------- -------- -------- --------
class HorizontalScrollBar : public SubWindow {
public:
	HorizontalScrollBar(const std::weak_ptr<SubWindow> &pparent);
};

// -------- -------- -------- -------- -------- -------- -------- --------
class VerticalScrollBar : public SubWindow {
public:
	VerticalScrollBar(const std::weak_ptr<SubWindow> &pparent);
};

// -------- -------- -------- -------- -------- -------- -------- --------
class ScrollWindow : public SubWindow{
public:
	ScrollWindow(const std::weak_ptr<SubWindow> &pparent);
private:
	std::shared_ptr<HorizontalScrollBar> _horizontalScrollBar;
	std::shared_ptr<VerticalScrollBar> _verticalScrollBar;
};

// -------- -------- -------- -------- -------- -------- -------- --------
class Frame : public SubWindow {
public:
	Frame(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
protected:
	void attributeFromJSON(const nlohmann::json &j);
private:
	int _borderWidth = 1;
	RGBA _borderColor = RGBA(255, 255, 255, 255);
	RGBA _backgroundColor = RGBA(63, 63, 63, 255);
};

// -------- -------- -------- -------- -------- -------- -------- --------
class GroupBox : public SubWindow {
public:
	GroupBox(const std::weak_ptr<SubWindow> &pparent)
		:SubWindow(pparent)
	{}
};

// -------- -------- -------- -------- -------- -------- -------- --------
class HorizontalLine : public SubWindow {
public:
	HorizontalLine(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
protected:
	void attributeFromJSON(const nlohmann::json &j);
private:
	int _width = 1;
};

// -------- -------- -------- -------- -------- -------- -------- --------
class VerticalLine : public SubWindow {
public:
	VerticalLine(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
protected:
	void attributeFromJSON(const nlohmann::json &j);
private:
	int _width = 1;
};


// -------- -------- -------- -------- -------- -------- -------- --------
class Label : public SubWindow {
public:
	Label(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
protected:
	IVec2 minSize();
	void attributeFromJSON(const nlohmann::json &j);
private:
	std::string _text;
};

// -------- -------- -------- -------- -------- -------- -------- --------
//
class PushButton : public SubWindow {
public:
	PushButton(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
	void onClick(std::function<void(void)> cb);
	void mouseLeftButtonDownEvent(int x, int y);
	void mouseEnterEvent();
	void mouseLeaveEvent();
protected:
	void attributeFromJSON(const nlohmann::json &j);
	IVec2 minSize();
private:
	std::function<void(void)> _onClick;
	std::string _text;
	bool _mouseIn = false;
};

// -------- -------- -------- -------- -------- -------- -------- --------
class RadioButton : public SubWindow {
public:
	RadioButton(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
	void onClick(std::function<void(bool)> cb);
	void mouseLeftButtonDownEvent(int x, int y);
	void attributeFromJSON(const nlohmann::json &j);
	void mouseEnterEvent();
	void mouseLeaveEvent();
private:
	std::function<void(bool)> _onClick;
	std::string _text;
	bool _value;
	bool _mouseIn = false;
};

// -------- -------- -------- -------- -------- -------- -------- --------
class CheckBox : public SubWindow {
public:
	CheckBox(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
	void onClick(std::function<void(bool)> cb);
	void mouseLeftButtonDownEvent(int x, int y);
	void attributeFromJSON(const nlohmann::json &j);
	void mouseEnterEvent();
	void mouseLeaveEvent();
private:
	std::function<void(bool)> _onClick;
	std::string _text;
	bool _value;
	bool _mouseIn = false;
};

// -------- -------- -------- -------- -------- -------- -------- --------
class HorizontalSlider : public SubWindow {
public:
	HorizontalSlider(const std::weak_ptr<SubWindow> &pparent);
	void paintEvent();
	//void mouseLeftButtonDownEvent(int x, int y);
	//void mouseLeftButtonUpEvent(int x, int y);
	//void mouseMoveEvent(int x, int y, int buttonState);
	//void mouseLeaveEvent();
	void attributeFromJSON(const nlohmann::json &j);
	void mouseEnterEvent();
	void mouseLeaveEvent();
public:
	void onValueChange();
	void valueChangeEvent();
private:
	int _value = 0;
	int _maxValue = 100;
	bool _drag = false;
	bool _mouseIn = false;
};

}; // end of namespace GLGUI
}; // end of namespace MOCHIMAZUI

#endif
