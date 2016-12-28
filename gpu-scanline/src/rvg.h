#ifndef _MOCHIMAZUI_RVG_H_
#define _MOCHIMAZUI_RVG_H_

#include <string>
#include <memory>

#include "vg_container.h"

namespace Mochimazui {

	using std::string;

	class RVG {

	public:
		void setA128(bool f) { _a128 = f; }
		void load(const string &fileName);

		int32_t width(){ return _viewport[1].x; }
		int32_t height() { return _viewport[1].y; }

		const std::shared_ptr<VGContainer> &vgContainer() const { return _spVGContainer; }

		void saveSelectedPath(const std::vector<uint32_t> &pids);

	private:

		bool _a128 = false;

		glm::ivec2 _viewport[2];
		glm::ivec2 _window[2];

		std::shared_ptr<VGContainer> _spVGContainer;

		std::string _header;
		std::vector<std::string> _lines;
	};
}

#endif
