
#ifndef _MOCHIMAZUI_SVG_H_
#define _MOCHIMAZUI_SVG_H_

#include <cstring>
#include <cstdint>

#include <map>
#include <vector>
#include <functional>
#include <memory>

#include <mochimazui/string.h>

#include <glm/glm.hpp>

#include "vg_container.h"

#include "gradient.h"

namespace Mochimazui {

	struct SVG {

	public: 
		//SVG();
		//~SVG();

	public:

		void setA128(bool f) { _a128 = f; }

		void load(const stdext::string &fileName, bool gen_nvpr_path_commands = false);
		void save(const stdext::string &fileName);

		const std::shared_ptr<VGContainer> &vgContainer() const { return _spVGContainer; }
		void setVg(std::shared_ptr<VGContainer> &pVg);

	public:
		uint32_t width() { return _width; }
		uint32_t height() { return _height; }

	private:

		bool _a128 = false;

		uint32_t _width = 0, _height = 0;
		glm::vec2 _viewBox[2];

		//std::vector<Gradient> _gradients;
		//std::map<std::string, uint32_t> _gradientMap;

		std::shared_ptr<VGContainer> _spVGContainer;
	};

}


#endif
