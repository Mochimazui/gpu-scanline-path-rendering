
#define  _CRT_SECURE_NO_WARNINGS

#include <mochimazui/glpp.h>

#include <cstdio>
#include <map>

#include <mochimazui/file.h>
#include <mochimazui/string.h>
#include <mochimazui/stdio_ext.h>

namespace Mochimazui {

namespace GLPP {

const std::string Shader::DefaultFileExtension(ShaderType t) {
	static const std::map<ShaderType, const std::string> type_to_file_ext = {
		{ ShaderType::Vertex, ".vert.glsl" },
		{ ShaderType::Geometry, ".geom.glsl" },
		{ ShaderType::Fragment, ".frag.glsl" },
	};
	auto i = type_to_file_ext.find(t);
	if (i == type_to_file_ext.end()) { throw std::runtime_error("Shader::DefaultFileExtension"); }
	return i->second;
}

Shader &Shader::define(const std::string &def) {
	_defines.push_back(def);
	return *this;
}

Shader &Shader::code(const std::string &code) { 

#ifndef _DEBUG
	if (_defines.empty() && _includes.empty()) {
		_code = code;
		return *this;
	}
#endif

	stdext::string ecode = code;
	auto lines = ecode.splitLine();

	_code.clear();
	_code.reserve(code.length());
	uint32_t i = 0;
	for (; i < lines.size(); ++i) {
		_code += lines[i] + "\n";
		if (lines[i].substr(0, 8) == "#version") {
			++i; break;
		}
	}

	uint32_t def_number = (uint32_t)_defines.size();
#ifdef _DEBUG
	++def_number;
	_code += "#define _DEBUG\n";
#endif
	for (auto def : _defines) {
		_code += "#define " + def + "\n";
	}

	char line_number[32];
	sprintf(line_number, "#line %u\n", i);
	_code += line_number;

	for (; i < lines.size(); ++i) {
		_code += lines[i] + "\n";
	}

	return *this; 
}

Shader &Shader::codeFromFile(const std::string &codeFileName) {
	// TODO: 
	//   include support ?
	//   use ARB_shading_language_include or process manually.
	//
	std::string code;
	readAll(codeFileName.c_str(), code);
	this->code(code);
	if (_name.empty()) { _name = codeFileName; }
	return *this;
}

void Shader::create() {
	if (!_shader) {
		_shader = glCreateShader(_type);
		if (!_shader) {
			printf("Error in Shader::gen: cannot create shader.\n");
		}
	}
}

void Shader::destroy() {
	if (_shader) { glDeleteShader(_shader); _shader = 0; }
}

void Shader::compile() {
	create();
	const GLchar *pc = _code.c_str();
	glShaderSource(_shader, 1, &pc, NULL);
	glCompileShader(_shader);
	glGetShaderiv(_shader, GL_COMPILE_STATUS, &_compiled);

	GLint size = 0;
	glGetShaderiv(_shader, GL_INFO_LOG_LENGTH, &size);
	if (size > 1) {
		char *log = new char[size];
		glGetShaderInfoLog(_shader, size, &size, log);
		if (!_compiled) { stdext::error_printf("Shader \"%s\"\n", _name.c_str()); }
		else { stdext::warning_printf("Shader \"%s\"\n", _name.c_str()); }
		printf("%s\n", log);
		delete[] log;
	}
}

// -------- -------- -------- -------- -------- -------- -------- --------
ShaderProgram::~ShaderProgram() {
	destroy();
}

ShaderProgram &ShaderProgram::create() {
	if (!_shaderProgram) { _shaderProgram = glCreateProgram(); }
	return *this;
}

void ShaderProgram::destroy() {
	if (_shaderProgram) {
		glDeleteProgram(_shaderProgram);
		_shaderProgram = 0;
	}
	for (auto &i : _shaders) {
		if (i.second) {
			i.second->destroy();
		}
	}
}

ShaderProgram &ShaderProgram::setShader(Shader &shader) {
	if (!shader.compiled()) {
		shader.compile();
	}
	create();
	glAttachShader(_shaderProgram, shader.shader());
	return *this;
}

ShaderProgram &ShaderProgram::setShader(std::shared_ptr<Shader> shader) {
	if (!shader->compiled()) {
		shader->compile();
	}
	_shaders[shader->type()] = shader;	
	create();
	glAttachShader(_shaderProgram, shader->shader());
	return *this;
}

ShaderProgram & ShaderProgram::createShader(ShaderType type, const std::string &code) {
	auto &pShader = _shaders[type];
	pShader.reset(new Shader(type));
	//
	pShader->code(code);
	pShader->compile();
	create();
	glAttachShader(_shaderProgram, pShader->shader());
	return *this;
}

ShaderProgram & ShaderProgram::createShaderFromFile(ShaderType type, const std::string &cf) {
	auto &pShader = _shaders[type];
	pShader.reset(new Shader(type));
	//
	pShader->codeFromFile(cf);
	pShader->compile();
	create();
	glAttachShader(_shaderProgram, pShader->shader());
	return *this;
}

void ShaderProgram::link() {
	create();
	glLinkProgram(_shaderProgram);
	glGetProgramiv(_shaderProgram, GL_LINK_STATUS, &_linked);
	GLint size = 0;
	glGetProgramiv(_shaderProgram, GL_INFO_LOG_LENGTH, &size);
	if (size > 1) {
		if (!_linked) { stdext::error_printf("ShaderProgram \"%s\"\n", _name.c_str()); }
		else { stdext::warning_printf("ShaderProgram \"%s\"\n", _name.c_str()); }
		char *log = new char[size];
		glGetProgramInfoLog(_shaderProgram, size, &size, log);
		printf("%s\n", log);
		delete[] log;
	}	
}

// -------- -------- -------- -------- -------- -------- -------- --------
void ShaderProgram::uniform1f(const char *name, GLfloat v) {
	glProgramUniform1f(_shaderProgram, location(name), v);
}

void ShaderProgram::uniform1i(const char *name, GLint v) {
	glProgramUniform1i(_shaderProgram, location(name), v);
}

void ShaderProgram::uniform1ui(const char *name, GLuint v) {
	glProgramUniform1ui(_shaderProgram, location(name), v);
}

void ShaderProgram::uniform2f(const char *name, GLfloat v0, GLfloat v1) {
	glProgramUniform2f(_shaderProgram, location(name), v0, v1);
}

void ShaderProgram::uniform2i(const char *name, GLint v0, GLint v1) {
	glProgramUniform2i(_shaderProgram, location(name), v0, v1);
}

void ShaderProgram::uniform2ui(const char *name, GLuint v0, GLuint v1) {
	glProgramUniform2ui(_shaderProgram, location(name), v0, v1);
}

void ShaderProgram::uniform3f(const char *name, GLfloat v0, GLfloat v1, GLfloat v2) {
	glProgramUniform3f(_shaderProgram, location(name), v0, v1, v2);
}

void ShaderProgram::uniform3i(const char *name, GLint v0, GLint v1, GLint v2) {
	glProgramUniform3i(_shaderProgram, location(name), v0, v1, v2);
}

void ShaderProgram::uniform3ui(const char *name, GLuint v0, GLuint v1, GLuint v2) {
	glProgramUniform3ui(_shaderProgram, location(name), v0, v1, v2);
}

void ShaderProgram::uniform4f(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
	glProgramUniform4f(_shaderProgram, location(name), v0, v1, v2, v3);
}

void ShaderProgram::uniform4i(const char *name, GLint v0, GLint v1, GLint v2, GLint v3) {
	glProgramUniform4i(_shaderProgram, location(name), v0, v1, v2, v3);
}

void ShaderProgram::uniform4ui(const char *name, GLuint v0, GLuint v1, GLuint v2, GLuint v3) {
	glProgramUniform4ui(_shaderProgram, location(name), v0, v1, v2, v3);
}

// -------- -------- -------- -------- -------- -------- -------- --------

void ShaderProgram::uniform1fv(const char *name, GLsizei count, GLfloat *pv) {
	glProgramUniform1fv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform1iv(const char *name, GLsizei count, GLint *pv) {
	glProgramUniform1iv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform1uiv(const char *name, GLsizei count, GLuint *pv) {
	glProgramUniform1uiv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform2fv(const char *name, GLsizei count, GLfloat *pv) {
	glProgramUniform2fv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform2iv(const char *name, GLsizei count, GLint *pv) {
	glProgramUniform2iv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform2uiv(const char *name, GLsizei count, GLuint *pv) {
	glProgramUniform2uiv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform3fv(const char *name, GLsizei count, GLfloat *pv) {
	glProgramUniform3fv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform3iv(const char *name, GLsizei count, GLint *pv) {
	glProgramUniform3iv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform3uiv(const char *name, GLsizei count, GLuint *pv) {
	glProgramUniform3uiv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform4fv(const char *name, GLsizei count, GLfloat *pv) {
	glProgramUniform4fv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform4iv(const char *name, GLsizei count, GLint *pv) {
	glProgramUniform4iv(_shaderProgram, location(name), count, pv);
}

void ShaderProgram::uniform4uiv(const char *name, GLsizei count, GLuint *pv) {
	glProgramUniform4uiv(_shaderProgram, location(name), count, pv);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void ShaderProgram::uniformMatrix2fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix2fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix2x3fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix2x3fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix2x4fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix2x4fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix3fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix3fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix3x2fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix3x2fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix3x4fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix3x4fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix4fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix4fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix4x2fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix4x2fv(_shaderProgram, location(name), count, transpose, value);
}

void ShaderProgram::uniformMatrix4x3fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value) {
	glProgramUniformMatrix4x3fv(_shaderProgram, location(name), count, transpose, value);
}

// -------- -------- -------- -------- -------- -------- -------- --------
void Util::init() {

	DEBUG_CHECK_GL_ERROR();

	_vertex_buffer.target(BufferTarget::Array).create();
	_color_buffer.target(BufferTarget::Array).create();

	_vertex_texture.target(TextureTarget::TextureBuffer).create();
	_color_texture.target(TextureTarget::TextureBuffer).create();

	_color_program.createShader(GLPP::Vertex,
		R"(
		#version 450

		uniform ivec2 vp_size;

		layout(binding = 0) uniform samplerBuffer tb_vertex;
		layout(binding = 1) uniform samplerBuffer tb_color;

		flat out vec4 fragColor;

		void main() {
			//
			vec4 draw = texelFetch(tb_vertex, gl_VertexID);
			vec2 p = vec2(draw.x / vp_size.x, draw.y / vp_size.y) * 2 - vec2(1.0, 1.0);
			fragColor = texelFetch(tb_color, gl_VertexID);
			gl_Position = vec4(p.x, p.y, 0, 1);
		}
		)");

	_color_program.createShader(GLPP::Fragment,
		R"(				
		#version 450

		in vec4 fragColor;

		layout(location = 0) out vec4 color;

		void main() {
			color = fragColor;
		}
		)");

	_color_program.link();

	DEBUG_CHECK_GL_ERROR();
}

void Util::viewport(int w, int h) {
	DEBUG_CHECK_GL_ERROR();
	_vp_x = _vp_y = 0;
	_vp_w = w; _vp_h = h;
	_color_program.uniform2i("vp_size", _vp_w, _vp_h);
	DEBUG_CHECK_GL_ERROR();
}

void Util::viewport(int x, int y, int w, int h) {
	DEBUG_CHECK_GL_ERROR();
	_vp_x = x; _vp_y = y;
	_vp_w = w; _vp_h = h;
	_color_program.uniform2i("vp_size", _vp_w, _vp_h);
	DEBUG_CHECK_GL_ERROR();
}

void Util::draw_lines( 
	const std::vector<float2> &v,
	const std::vector<frgba> &c
	) {

	DEBUG_CHECK_GL_ERROR();

	_vertex_buffer.data((GLsizei)v.size() * sizeof(float2), v.data(), GL_STATIC_DRAW);
	_color_buffer.data((GLsizei)c.size() * sizeof(frgba), c.data(), GL_STATIC_DRAW);

	_vertex_texture.buffer(GL_RG32F, _vertex_buffer).bindUnit(0);
	_color_texture.buffer(GL_RGBA32F, _color_buffer).bindUnit(1);

	_color_program.use();

	glLineWidth(1.f);
	glDrawArrays(GL_LINES, 0, (GLsizei)v.size());

	_color_program.disuse();

	DEBUG_CHECK_GL_ERROR();
}

//void Util::draw_lines(const float2* v, int n) {
//	_color_program.use();
//	glDrawArrays(GL_LINES, 0, n);
//}

} // end of namespace GLPP

} // end of namespace Mochimazui
