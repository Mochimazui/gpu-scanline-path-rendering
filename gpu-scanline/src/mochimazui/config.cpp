
#include <mochimazui/config.h>
#include <mochimazui/string.h>
#include <mochimazui/file.h>

namespace Mochimazui {

using namespace ConfigPrivate;

using std::vector;
using std::string;
using std::basic_string;
using std::runtime_error;

namespace program_options = boost::program_options;

// -------- -------- -------- -------- -------- -------- -------- --------
void Config::addValue(const std::string &iv) {
	stdext::string v = iv;
	auto l = v.split(':');
	if (l.size() != 2 && l.size() != 3) {
		throw std::runtime_error("Config::addValue: invalid ConfigValue format " + iv);
	}
	ConfigValue cv;
	cv.setName(l[0]);
	cv.setType(l[1]);
	if (l.size() == 3) {
		cv.setValue(l[2].c_str());
	}
	_value_map[l[0]] = cv;
}

void Config::addValue(const std::vector<std::string> &vv) {
	for (const auto &v : vv) {
		addValue(v);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void Config::addOption(const std::string &o) {

	ConfigOption co;

	auto i_space = o.find(' ');

	co.name = o.substr(0, i_space);

	auto i_lb = o.find('{');
	auto i_rb = o.find('}');

	// help info, ignore.

	stdext::string values = o.substr(i_rb + 2);
	auto value_list = values.split('#');

	if (value_list[0] == "0") { co.type = ConfigOption_Void; }
	else if (value_list[0] == "1") { co.type = ConfigOption_Value; }
	else if (value_list[0] == "*") { co.type = ConfigOption_Array; }
	else { throw std::runtime_error("Config::addOption: invalid option format."); }

	for (int i = 1; i < value_list.size(); ++i) {
		auto l = value_list[i].split(':');
		if (l.size() != 1 && l.size() != 2) {
			throw std::runtime_error("Config::addOption: invalid option format.");
		}
		ConfigOptionSetValue sv;
		sv.name = l[0];
		if (l.size() == 2) { sv.value = l[1]; }
		co.values.push_back(sv);
	}

	_option_map["-" + co.name] = co;
}

void Config::addOption(const std::vector<std::string> &vo) {
	for (const auto &o : vo) {
		addOption(o);
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
void help() {
}

// -------- -------- -------- -------- -------- -------- -------- --------
void Config::load(const std::string &file) {

	vector<string> args;
	command_line_file_to_args(file, args);

	vector<const char*> args_ptr;
	for (string &s : args) { args_ptr.push_back(s.data()); }

	parse((int)args_ptr.size(), args_ptr.data());
}

void Config::parse(int argc, const char *argv[]) {

	for (int i = 1; i < argc;) {
		std::string arg = argv[i];
		auto ioption = _option_map.find(arg);
		if (ioption == _option_map.end()) {
			throw std::runtime_error("Config::parse: unsupported option " + arg);
		}
		++i;

		const auto &co = ioption->second;

		std::string value;
		if (co.type != ConfigOption_Void) { 
			if (i >= argc) { throw std::runtime_error(arg + " requires more input."); }
			value = argv[i];
			++i;
		}

		for (const auto &v : co.values) {
			auto iv = _value_map.find(v.name);
			if (iv == _value_map.end()) {
				ConfigValue new_value;
				new_value.setName(v.name);
				new_value.setType("any");
				new_value.setValue(v.value.empty() ? value : v.value);
			}
			else {
				iv->second.setValue(v.value.empty() ? value : v.value);
			}
		}

	}

}


}
