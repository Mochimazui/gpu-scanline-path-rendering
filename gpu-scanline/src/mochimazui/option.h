
#ifndef _MOCHIMAZUI_OPTION_H_
#define _MOCHIMAZUI_OPTION_H_

#include <vector>
#include <unordered_map>
#include "string.h"

namespace Mochimazui {

enum OptionType {
	Int,
	Float,
	String,

	IntArray,
	FloatArray,
	StringArray,
};

class OptionInfo {
	std::string name;
	std::string shortcut;
	OptionType valueType;
	std::string value;
};

template <class T>
class OptionWithPointer {
};

template <class T>
class OptionWithReference {
};

// -------- -------- -------- -------- -------- -------- -------- --------
class Option {

public:
	Option &addOption(const std::string &name, OptionType type) {
		return *this;
	}

	template <class T>
	Option &addOption(const std::string &name, OptionType type, T*) {
		return *this;
	}

	template <class T>
	Option &addOption(const std::string &name, OptionType type, T&) {
		return *this;
	}

	//Option &addOption(const std::string &name, OptionType type, ) {}

public:
	Option &operator()(int argc, char *argv[]) {}
	Option &operator()(const std::string &fileName) {}

private:

};

}

#endif
