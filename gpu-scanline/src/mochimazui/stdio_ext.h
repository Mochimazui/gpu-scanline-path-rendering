
#pragma once

#ifndef _MOCHIMAZUI_STDIO_EXT_
#define _MOCHIMAZUI_STDIO_EXT_

#if defined _WIN32 || defined _WIN64
#define MOCHIMAZUI_WINDOWS
#include <Windows.h>
#undef min
#undef max
#endif

namespace Mochimazui {
namespace stdext {

namespace stdext_private {
#ifdef MOCHIMAZUI_WINDOWS
inline HANDLE console_handle() {
	static HANDLE s_h_console = GetStdHandle(STD_OUTPUT_HANDLE);
	return s_h_console;
}
inline void set_console_text_attribute(WORD a) {
	//#define FOREGROUND_BLUE      0x0001 // text color contains blue.
	//#define FOREGROUND_GREEN     0x0002 // text color contains green.
	//#define FOREGROUND_RED       0x0004 // text color contains red.
	//#define FOREGROUND_INTENSITY 0x0008 // text color is intensified.
	//#define BACKGROUND_BLUE      0x0010 // background color contains blue.
	//#define BACKGROUND_GREEN     0x0020 // background color contains green.
	//#define BACKGROUND_RED       0x0040 // background color contains red.
	//#define BACKGROUND_INTENSITY 0x0080 // background color is intensified.
	SetConsoleTextAttribute(console_handle(), a);
}
#else
inline void set_console_text_attribute(uint32_t) {
}
#endif
}

template<typename ...Ts> 
inline void color_printf(int text_color, int background_color, const char *fmt_str, Ts... args) {
	stdext_private::set_console_text_attribute((background_color << 4) | text_color);
	printf(fmt_str, args...);
	stdext_private::set_console_text_attribute(7);
}

template<typename ...Ts>
inline void error_printf(const char *fmt_str, Ts... args) {
	stdext_private::set_console_text_attribute((12 << 4) | 15);
	fprintf(stderr, fmt_str, args...);
	stdext_private::set_console_text_attribute(7);
}

template<typename ...Ts>
inline void warning_printf(const char *fmt_str, Ts... args) {
	stdext_private::set_console_text_attribute((6 << 4) | 15);
	fprintf(stderr, fmt_str, args...);
	stdext_private::set_console_text_attribute(7);
}

template<typename ...Ts>
inline void info_printf(const char *fmt_str, Ts... args) {
	stdext_private::set_console_text_attribute((10 << 4) | 15);
	fprintf(stderr, fmt_str, args...);
	stdext_private::set_console_text_attribute(7);
}

}
}

#endif