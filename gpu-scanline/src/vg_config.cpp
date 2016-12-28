
#define _CRT_SECURE_NO_WARNINGS

#include "vg_config.h"

#include <cstring>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>

#include <mochimazui/file.h>
#include <mochimazui/string.h>
#include <mochimazui/stdio_ext.h>

// -------- -------- -------- -------- -------- -------- -------- --------
namespace Mochimazui {

namespace PRIVATE {
boost::program_options::variables_map g_config_variables;
}

int init_config(int argc, char *argv[]) {

	namespace po = boost::program_options;

	using PRIVATE::g_config_variables;
	using Mochimazui::parse_command_line_file;

	po::options_description general_options("General options");
	general_options.add_options()
		("help", "print help")

		("verbose", po::bool_switch(), "verbose output to console")
		("gl-debug", po::bool_switch(), "enable GL_DEBUG")
		("draw-curve", po::bool_switch(), "draw curve")
		("show-fps", po::bool_switch(), "show fps")

		("benchmark", po::bool_switch(), "benchmark")
		("step-timing", po::bool_switch(), "step timing")
		("attach-timing-to", po::value<std::string>()->default_value(""), "")

		("merge-path", po::bool_switch(), "")
		("minimal-ui", po::bool_switch(), "produce help message")

		("v-flip", po::bool_switch(), "")

		("count-pixel", po::bool_switch(), "")
		("attach-pixel-count-to", po::value<std::string>()->default_value(""), "")

		("animation", po::bool_switch(), "run chrod animation")
		;

	po::options_description io_options("Input/output options");
	io_options.add_options()
		("file-index", po::value<std::string>()->multitoken(), "file index")

		("input-name", po::value<std::string>()->default_value(""), "")
		("input-file,i", po::value<std::string>(), "input file")

		("input-width", po::value<int>()->default_value(0), "")
		("input-height", po::value<int>()->default_value(0), "")

		("window-width", po::value<int>()->default_value(1200), "")
		("window-height", po::value<int>()->default_value(1024), "")

		("fit-to-vg", po::bool_switch(), "")
		("fit-to-window", po::bool_switch(), "")

		("save-output-file", po::bool_switch(), "")
		("output-file,o", po::value<std::string>()->default_value(""), "input file")

		("output-width", po::value<int>()->default_value(0), "")
		("output-height", po::value<int>()->default_value(0), "")
		("fix-output-size", po::bool_switch(), "")
		;

	po::options_description rasterizer_options("Rasterizer options");
	rasterizer_options.add_options()
		("c-m-cs", po::bool_switch(), "cut, mask table, comb-like scanline")

		("lrgb", po::bool_switch(), "")
		("srgb", po::bool_switch(), "")

		("samples", po::value<int>()->default_value(32), "")
		("ms-output", po::bool_switch(), "")

		("reserve-ink", po::value<int>()->default_value(0), "reserve ink")
		("tiger-clip", po::bool_switch(), "")

		("break-before-gl", po::bool_switch(), "break before gl step")

		("a128", po::bool_switch(), "align alpha value to 1/128")
		;

	po::options_description all_options;
	all_options.add(general_options).add(io_options).add(rasterizer_options);

	if (argc == 1) {	
		po::store(parse_command_line_file<char>("vg_default.cfg", all_options), g_config_variables);
	}
	else {
		po::store(po::parse_command_line(argc, argv, all_options), g_config_variables);
	}

	po::notify(g_config_variables);

	if (g_config_variables.count("help")) {
		printf("\nLoads \"vg_default.cfg\" by default.\n");
		printf("Using command line argument will skip config file loading.\n");
		std::cout << all_options << "\n";
		return -1;
	}

	return 0;

}

}
