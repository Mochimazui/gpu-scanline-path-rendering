
#ifndef _MOCHIMAZUI_GRADIENT_H_
#define _MOCHIMAZUI_GRADIENT_H_

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <mochimazui/color.h>

namespace Mochimazui {

	enum GradientType {
		GT_Linear = 2, 
		GT_Radial = 3
	};

	enum GradientUnits {
		USER_SPACE_ON_USE,
		OBJECT_BOUNDING_BOX
	};

	struct GradientStop {
		float offset;
		u8rgba color;
		float opacity = 1.f;
	};

	// Gradient-related enumerations
	enum SpreadMethod {
		PAD,      // clamp to edge
		REFLECT,  // mirror
		REPEAT,   // repeat
		NONE      // clamp to border with (0,0,0,0) border
	};

	inline bool operator < (const GradientStop &a, const GradientStop &b) {
		return a.offset < b.offset;
	}

	struct Gradient {

		GradientType gradient_type;
		GradientUnits gradient_units = USER_SPACE_ON_USE;
		glm::mat3x3 gradient_transform;  // could be float4x4
		SpreadMethod spread_method = PAD;
		std::vector<GradientStop> gradient_stops;

		std::string href;

		// Linear gradient attributes
		glm::vec2 v1, v2;

		// Radial gradient attributes
		glm::vec2 c;  // center
		glm::vec2 f;  // focal point
		float r;   // radius

		bool f_set = false;

		void clear() {
			gradient_units = USER_SPACE_ON_USE;
			gradient_transform = glm::mat3x3();
			spread_method = PAD;
			gradient_stops.clear();
			href.clear();

			f_set = false;
			v1 = v2 = c = f = glm::vec2();
			r = 0.f;
		}

	};

}

#endif
