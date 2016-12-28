
#ifndef _MOCHIMAZUI_TIGER_H_
#define _MOCHIMAZUI_TIGER_H_

#include <mochimazui/3rd/gl_4_5_core.h>

namespace Mochimazui {

	namespace Tiger {

		struct TigerStyle {
			GLuint fill_color;
			GLuint stroke_color;
			GLfloat stroke_width;
		};

		extern const char *tiger_path[240];
		extern const TigerStyle tiger_style[240];

		extern const unsigned int tiger_path_count;
		extern GLuint tiger_path_base;

		void initTiger();
		void drawTiger(int filling, int stroking);

	}
}


#endif
