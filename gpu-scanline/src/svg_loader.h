
#pragma once

#include <string>
#include <memory>

namespace Mochimazui {

class VGContainer;

std::shared_ptr<VGContainer> 
load_svg(const std::string &file_name, bool stroke_to_fill);

}
