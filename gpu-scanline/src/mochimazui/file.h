
#ifndef _MOCHIMAZUI_FILE_H_
#define _MOCHIMAZUI_FILE_H_

#include <string>
#include <vector>
#include <array>

#include <mochimazui/stdio_ext.h>

namespace Mochimazui {

template<class charT>
void readAll(const charT *fn, std::basic_string<charT> &odata) {

	FILE *fin;
	fin = fopen(fn, "rb");
	if (!fin) {
		auto msg = "Error in readAll: can not open file \"" + std::string(fn) + "\"";
		stdext::error_printf("%s", fn);
		throw std::runtime_error(msg);
	}

	fseek(fin, 0, SEEK_END);
	long size = ftell(fin);

	charT *data = new charT[size + 1];
	if (!data) { printf("Error in readAll: new char returnd 0\n"); return; }

	fseek(fin, 0, SEEK_SET);
	size_t size_read = fread(data, 1, size, fin);
	fclose(fin);

	data[size] = '\0';
	odata = data;
}

}

#endif