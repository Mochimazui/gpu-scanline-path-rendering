
#pragma once

#ifndef _MOCHIMAZUI_CONFIG_H_
#define _MOCHIMAZUI_CONFIG_H_

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#define MOCHIMAZUI_UNDEFINE_CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <initializer_list>
#include <string>
#include <vector>
#include <map>

#include <boost/program_options.hpp>

namespace Mochimazui {

// -------- -------- -------- -------- -------- -------- -------- --------
template<class charT>
void command_line_file_to_args(
	const std::basic_string<charT> &argv0,
	const std::basic_string<charT> &fn,
	std::vector<std::basic_string<charT>> &args) {

	typedef std::basic_string<charT> string_t;

	// argv[0]
	args.push_back(argv0);

	// read file
	string_t file_text;
	readAll(fn.c_str(), file_text);

	// command line file -> command line options

	string_t current;
	charT last_c = '\0';
	bool in_string = false;
	bool in_comment = false;
	bool escape_char = false;

	// allow multiline string surrounded by ""
	// allow \" and \\ in string
	// allow single line comment startting with // or #

	for (auto c : file_text) {
		switch (c) {
		case '\n':
		case '\r':
			if (in_comment) {
				in_comment = false;
				break;
			}
			// no break here
		case ' ':
			if (in_comment) { break; }
			if (in_string) {
				current.push_back(c);
			}
			if (!current.empty()) {
				args.push_back(current);
				current.clear();
			}
			break;
		case '"':
			if (in_comment) { break; }
			if (in_string) {
				if (escape_char) {
					current.push_back('"');
					escape_char = false;
				}
				else {
					in_string = false;
				}
			}
			else {
				in_string = true;
			}
			break;
		case '\\':
			if (in_comment) { break; }
			if (in_string) {
				if (escape_char) {
					current.push_back('\\');
					escape_char = false;
				}
				else {
					escape_char = true;
				}
			}
			else {
				current.push_back('\\');
			}
			break;
		case '#':
			if (in_comment) { break; }
			if (!in_string) { in_comment = true; }
			break;
		case '/':
			if (in_comment) { break; }
			if (!in_string && last_c == '/') {
				current.pop_back();
				in_comment = true;
			}
			else {
				current.push_back(c);
			}
			break;
		default:
			if (in_comment) { break; }
			current.push_back(c);
			break;
		}
		last_c = c;
	}
}

template<class charT>
void command_line_file_to_args(
	const std::basic_string<charT> &fn,
	std::vector<std::basic_string<charT>> &args) {
	command_line_file_to_args(std::basic_string<charT>(), fn, args);
}

// -------- -------- -------- -------- -------- -------- -------- --------
template<class charT>
boost::program_options::basic_parsed_options<charT>
parse_command_line_file(
	const std::basic_string<charT> &fn,
	const boost::program_options::options_description& desc) {

	std::vector<std::basic_string<charT>> args;
	command_line_file_to_args(fn, args);

	std::vector<const charT*> args_ptr;
	for (auto &s : args) { args_ptr.push_back(s.data()); }
	return boost::program_options::parse_command_line(
		(int)args_ptr.size(), args_ptr.data(), desc);
}

// -------- -------- -------- -------- -------- -------- -------- --------
enum ConfigValueType {
	ConfigValue_Int32,
	ConfigValue_Int64,
	ConfigValue_Float,
	ConfigValue_Double,
	ConfigValue_String,
	ConfigValue_Bool,
	ConfigValue_Array,
	ConfigValue_Any,
};

class ConfigValueArray;

class ConfigValue {

public:
	void setName(const std::string &n) { name = n; }
	void setType(ConfigValueType t) {
		type = t;
	}
	void setType(const std::string &t) {
		static const std::map<std::string, ConfigValueType> tmap = {
			{ "int", ConfigValue_Int32 },
			{ "int32", ConfigValue_Int32 },
			{ "int64", ConfigValue_Int64 },
			{ "float", ConfigValue_Float },
			{ "double", ConfigValue_Double },
			{ "string", ConfigValue_String },
			{ "bool", ConfigValue_Bool },
			{ "array", ConfigValue_Array },
			{ "any", ConfigValue_Any },
		};
		type = tmap.find(t)->second;
	}

	void setValue(const std::string &v) {

		string_value = v;
		switch (type) {
		case ConfigValue_Int32:
			sscanf(v.c_str(), "%d", &uvalue.i32_value);
			break;
		case ConfigValue_Int64:
			sscanf(v.c_str(), "%ld", &uvalue.i64_value);
			break;
		case ConfigValue_Float:
			sscanf(v.c_str(), "%f", &uvalue.float_value);
			break;
		case ConfigValue_Double:
			sscanf(v.c_str(), "%lf", &uvalue.float_value);
			break;
		case ConfigValue_Bool:
			uvalue.bool_value = (v == "true");
			break;
		}

	}

public:
	int32_t toInt32() const { 
		if (type != ConfigValue_Int32) { throw_type_error(__FILE__, __LINE__); }
		return uvalue.i32_value; 
	}

	int64_t toInt64() const {
		if (type != ConfigValue_Int64) { throw std::runtime_error(""); }
		return uvalue.i32_value;
	}

	float toFloat() const {
		if (type != ConfigValue_Float) { throw std::runtime_error(""); }
		return uvalue.float_value;
	}

	double toDouble() const {
		if (type != ConfigValue_Double) { throw std::runtime_error(""); }
		return uvalue.double_value;
	}

	bool toBool() const {
		if (type != ConfigValue_Bool) { throw std::runtime_error(""); }
		return uvalue.bool_value;
	}

	const std::string &toString() const {
		return string_value;
	}

	ConfigValueArray toArray() const;

private:
	void throw_type_error(char *file, int line) const {
#ifdef _DEBUG
#else
#endif
	}

private:
	std::string name;
	ConfigValueType type = ConfigValue_String;
	union {
		int32_t i32_value;
		int64_t i64_value;
		float float_value;
		double double_value;
		bool bool_value;
	} uvalue;
	std::string string_value;
};

class ConfigValueArray : public std::vector<ConfigValueArray> {

public:
	ConfigValueArray() {}
	ConfigValueArray(const std::string &) {
	}
	~ConfigValueArray() {}

};

inline ConfigValueArray ConfigValue::toArray() const {
	if (type != ConfigValue_Array) { throw_type_error(__FILE__, __LINE__); }
	return ConfigValueArray(string_value);
}

namespace ConfigPrivate {

template<class T>
inline T get(const ConfigValue &v) {
	throw std::runtime_error("");
}

template<>
inline int32_t get<int32_t>(const ConfigValue &v) { return v.toInt32(); }

template<>
inline int64_t get<int64_t>(const ConfigValue &v) { return v.toInt64(); }

template<>
inline float get<float>(const ConfigValue &v) { return v.toFloat(); }

template<>
inline double get<double>(const ConfigValue &v) { return v.toDouble(); }

template<>
inline bool get<bool>(const ConfigValue &v) { return v.toBool(); }

template<>
inline std::string get<std::string>(const ConfigValue &v) { return v.toString(); }

template<>
inline ConfigValueArray get<ConfigValueArray>(const ConfigValue &v) { return v.toArray(); }

enum ConfigOptionType {
	ConfigOption_Void,
	ConfigOption_Value,
	ConfigOption_Array,
};

struct ConfigOptionSetValue {
	std::string name;
	std::string value;
};

struct ConfigOption {
	std::string name;
	ConfigOptionType type;
	std::vector<ConfigOptionSetValue> values;
};

}

class Config {

public:

	// -------- -------- -------- --------
	// value-name:type:default-value
	//
	void addValue(const std::string &v);
	void addValue(const std::vector<std::string> &vv);

	// -------- -------- -------- --------
	// name {help-info}  (v:a)#(value-name:value)+
	//
	// format-string:
	//   option-name#(value-name:value)+
	//   
	//   if option-name is empty, this option does not take any input.
	//
	//   empty option-name means it doesn't require a type.
	//
	// example:
	// addOption("-width -w {window width} <width:int@width:>")
	// addOption("-V {verbose} <:bool@verbose:true>")

	void addOption(const std::string &o);
	void addOption(const std::vector<std::string> &vo);

	// -------- -------- -------- --------
	void help();

	// -------- -------- -------- --------
	void load(const std::string &file);
	void parse(int argc, const char *argv[]);

	// -------- -------- -------- --------
	void clear();

	// -------- -------- -------- --------
	template<class T>
	T get(const std::string &name) {
		auto iv = _value_map.find(name);
		if (iv == _value_map.end()) {
			throw std::runtime_error("Config::get: value \"" + name + "\" not found.");
		}
		return ConfigPrivate::get<T>(_value_map[name]);
	}

private:
	std::map<std::string, ConfigPrivate::ConfigOption> _option_map;
	std::map<std::string, ConfigValue> _value_map;
};

} // end of namespace Mochimazui

#ifdef MOCHIMAZUI_UNDEFINE_CRT_SECURE_NO_WARNINGS
#undef _CRT_SECURE_NO_WARNINGS
#undef MOCHIMAZUI_UNDEFINE_CRT_SECURE_NO_WARNINGS
#endif

#endif
