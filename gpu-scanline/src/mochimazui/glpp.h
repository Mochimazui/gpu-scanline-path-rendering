
#ifndef _MOCHIMAZUI_GLPP_H_
#define _MOCHIMAZUI_GLPP_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <mochimazui/3rd/gl_4_5_compatibility.h>
#include <GL/GLU.h>

#include <mochimazui/color.h>

// FIXME: use CUDA's vector_types before find better soluition.
#include <vector_types.h>

namespace Mochimazui {

namespace GLPP {

// -------- -------- -------- -------- -------- -------- -------- --------
inline void checkGLError() {
	glFinish();
	auto e = glGetError();
	if (e != GL_NO_ERROR) {
		auto estr = gluErrorString(e);
		printf("%s\n", estr);
		throw std::runtime_error((char*)estr);
	}
}

#define CHECK_GL_ERROR() Mochimazui::GLPP::checkGLError()
#ifdef _DEBUG
#define DEBUG_CHECK_GL_ERROR() Mochimazui::GLPP::checkGLError()
#else
#define DEBUG_CHECK_GL_ERROR()
#endif

// -------- -------- -------- -------- -------- -------- -------- --------
class GLResource {
public:
	virtual void create() = 0;
	virtual void destroy() = 0;
private:
};

class GLResourceManager {
public:
	GLResourceManager& add(GLResource *pr) {
		_resources.push_back(pr);
		return *this;
	}
	void create() {
		for (auto p : _resources) { p->create(); }
	}
	void destroy() {
		for (auto p : _resources) { p->destroy(); }
	}
private:
	std::vector<GLResource*> _resources;
};

// -------- -------- -------- -------- -------- -------- -------- --------
enum ShaderType {
	Null,
	Vertex = GL_VERTEX_SHADER,
	Geometry = GL_GEOMETRY_SHADER,
	TessControl = GL_TESS_CONTROL_SHADER,
	TessEvaluation = GL_TESS_EVALUATION_SHADER,
	Fragment = GL_FRAGMENT_SHADER,
	Compute = GL_COMPUTE_SHADER,
};

class Shader {
public:
	Shader() {}
	Shader(ShaderType t) : _type(t) {}
	~Shader() { destroy(); }
public:
	// set
	Shader &name(const std::string &name) { _name = name; return *this; }
	Shader &define(const std::string &def);
	Shader &code(const std::string &code);
	Shader &codeFromFile(const std::string &cf);
	Shader &type(ShaderType type) { _type = type; return *this; }	
	// get
	const std::string &name() { return _name; }
	const std::string &code() const { return _code; }
	ShaderType type() { return _type; }
	GLuint shader() { return _shader; }
	bool compiled() { return !!_compiled; }
	//
	void create();
	void destroy();
	void compile();
public:
	static const std::string DefaultFileExtension(ShaderType t);
private:
	std::string _name;

	ShaderType _type = Null;
	std::string _code;
	GLuint _shader = 0;
	GLint _compiled = 0;

	std::vector<std::string> _defines;
	std::vector<std::string> _includes;
};

class ShaderProgram {
public:
	~ShaderProgram();
public:
	ShaderProgram &setShader(Shader &shader);
	ShaderProgram &setShader(std::shared_ptr<Shader> sp_shader);
	//ShaderProgram &setShader(const std::string &code, ShaderType type);
	//ShaderProgram &setShaderFile(const std::string &cf, ShaderType type);

	ShaderProgram &createShader(ShaderType type, const std::string &code);
	ShaderProgram &createShaderFromFile(ShaderType type, const std::string &code);

	ShaderProgram &create();
	void destroy();

	ShaderProgram &name(const std::string &name) { _name = name; return *this; }
	const std::string &name() { return _name; }

	void link();
	bool linked() { return !!_linked; }

	void use() { glUseProgram(_shaderProgram); }
	void disuse() { glUseProgram(0); }

	GLuint program() { return _shaderProgram; }
	operator GLuint() { return _shaderProgram; }

public:

	void uniform1f(const char *name, GLfloat v);
	void uniform1i(const char *name, GLint v);
	void uniform1ui(const char *name, GLuint v);

	void uniform2f(const char *name, GLfloat v0, GLfloat v1);
	void uniform2i(const char *name, GLint v0, GLint v1);
	void uniform2ui(const char *name, GLuint v0, GLuint v1);

	void uniform3f(const char *name, GLfloat v0, GLfloat v1, GLfloat v2);
	void uniform3i(const char *name, GLint v0, GLint v1, GLint v2);
	void uniform3ui(const char *name, GLuint v0, GLuint v1, GLuint v2);

	void uniform4f(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
	void uniform4i(const char *name, GLint v0, GLint v1, GLint v2, GLint v3);
	void uniform4ui(const char *name, GLuint v0, GLuint v1, GLuint v2, GLuint v3);

	void uniform1fv(const char *name, GLsizei count, GLfloat *pv);
	void uniform1iv(const char *name, GLsizei count, GLint *pv);
	void uniform1uiv(const char *name, GLsizei count, GLuint *pv);
	
	void uniform2fv(const char *name, GLsizei count, GLfloat *pv);
	void uniform2iv(const char *name, GLsizei count, GLint *pv);
	void uniform2uiv(const char *name, GLsizei count, GLuint *pv);
	
	void uniform3fv(const char *name, GLsizei count, GLfloat *pv);
	void uniform3iv(const char *name, GLsizei count, GLint *pv);
	void uniform3uiv(const char *name, GLsizei count, GLuint *pv);

	void uniform4fv(const char *name, GLsizei count, GLfloat *pv);
	void uniform4iv(const char *name, GLsizei count, GLint *pv);
	void uniform4uiv(const char *name, GLsizei count, GLuint *pv);

	void uniformMatrix2fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix2x3fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix2x4fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix3fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix3x2fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix3x4fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix4fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix4x2fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);
	void uniformMatrix4x3fv(const char *name, GLsizei count, GLboolean transpose, const GLfloat *value);

private:
	GLint location(const char *name) { return glGetUniformLocation(_shaderProgram, name); }

private:
	std::string _name;
	GLuint _shaderProgram = 0;
	GLint _linked = 0;
	std::unordered_map<ShaderType, std::shared_ptr<Shader>> _shaders;
};

// -------- -------- -------- -------- -------- -------- -------- --------
class VertexArray {
public:
	void create() {
		if (!_vertexArray) { glCreateVertexArrays(1, &_vertexArray); }
	}
	void destroy() {
		if (_vertexArray) {
			glDeleteVertexArrays(1, &_vertexArray); _vertexArray = 0;
		}
	}
	void bind() {
		glBindVertexArray(_vertexArray);
	}
	void unbind() {
		glBindVertexArray(0);
	}
private:
	GLuint _vertexArray = 0;
};

// -------- -------- -------- -------- -------- -------- -------- --------
enum BufferTarget {
	Array = GL_ARRAY_BUFFER,
	AtomicCounter = GL_ATOMIC_COUNTER_BUFFER,
	CopyRead = GL_COPY_READ_BUFFER,
	CopyWrite = GL_COPY_WRITE_BUFFER,
	DisplayIndirect = GL_DISPATCH_INDIRECT_BUFFER,
	DrawIndirect = GL_DRAW_INDIRECT_BUFFER,
	ElementArray = GL_ELEMENT_ARRAY_BUFFER,
	PixelPack = GL_PIXEL_PACK_BUFFER,
	PixleUnpack = GL_PIXEL_UNPACK_BUFFER,
	Query = GL_QUERY_BUFFER,
	ShaderStorage = GL_SHADER_STORAGE_BUFFER,
	Texture = GL_TEXTURE_BUFFER,
	TransformFeedback = GL_TRANSFORM_FEEDBACK_BUFFER,
	Uniform = GL_UNIFORM_BUFFER,
};

class Buffer {
public:
	Buffer() {}
	Buffer(BufferTarget target) :_target(target) {}
	Buffer &target(BufferTarget target) { _target = target; return *this; }
	BufferTarget target() { return _target; }
	Buffer &gen() { if (!_buffer) { glGenBuffers(1, &_buffer); } return *this; }
	void del() { glDeleteBuffers(1, &_buffer); _buffer = 0; }
	Buffer &bind() { glBindBuffer(_target, _buffer); return *this; }
	Buffer &unbind() { glBindBuffer(_target, 0); return *this; }
	Buffer &data(GLsizeiptr size, const void * data, GLenum usage) {
		bind();
		glBufferData(_target, size, data, usage);
		unbind();
		return *this;
	}
private:
	GLuint _buffer = 0;
	BufferTarget _target = Array;
};

class NamedBuffer {
public:
	NamedBuffer() {}
	NamedBuffer(BufferTarget target) :_target(target) {}
	NamedBuffer & create() {
		if (!_buffer) { glCreateBuffers(1, &_buffer); }
		return *this;
	}
	NamedBuffer &destroy() { glDeleteBuffers(1, &_buffer); _buffer = 0; return *this; }
	size_t size() const { return _size; }
	NamedBuffer &target(BufferTarget target) { _target = target; return *this; }
	void *map(GLenum access) {
		return glMapNamedBuffer(_buffer, access);
	}
	void *mapRange(GLintptr offset, GLsizei length, GLbitfield access) {
		return glMapNamedBufferRange(_buffer, offset, length, access);
	}
	void unmap() {
		glUnmapNamedBuffer(_buffer);
	}
	void storage(GLsizei size, const void *data, GLbitfield flags) {
		_size = size;
		glNamedBufferStorage(_buffer, size, data, flags);
	}
	void data(GLsizei size, const void *data, GLenum usage) {
		_size = size;
		glNamedBufferData(_buffer, size, data, usage);
	}
	void subData(GLintptr offset, GLsizei size, const void *data) {
		glNamedBufferSubData(_buffer, offset, size, data);
	}
	GLuint buffer() { return _buffer; }
	operator GLuint() { return _buffer; }
private:
	GLuint _buffer = 0;
	BufferTarget _target = Array;
	size_t _size = 0;
};

enum TextureTarget {
	Texture2D = GL_TEXTURE_2D,
	Texture2DMultisample = GL_TEXTURE_2D_MULTISAMPLE,
	TextureBuffer = GL_TEXTURE_BUFFER
};

class Texture {
public:
private:
};

class NamedTexture {
public:
	NamedTexture &target(TextureTarget target) { _target = target;  return *this; }
	NamedTexture &create() {
		if (!_texture) {
			glCreateTextures(_target, 1, &_texture);
		} return *this;
	}
	NamedTexture &destroy() { 
		if (_texture) { glDeleteTextures(1, &_texture); _texture = 0; }
		return *this;
	}
	NamedTexture &storage1D(GLsizei levels, GLenum internalformat, GLsizei width) {
		glTextureStorage1D(_texture, levels, internalformat, width);
		return *this;
	}
	NamedTexture &storage2D(GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) {
		glTextureStorage2D(_texture, levels, internalformat, width, height);
		return *this;
	}
	NamedTexture &storage2DMultisample(GLsizei samples, GLenum internalformat, 
		GLsizei width, GLsizei height, GLboolean fixedsamplelocations) {
		glTextureStorage2DMultisample(_texture, samples, internalformat, width, height, 
			fixedsamplelocations); return *this;
	}
	NamedTexture &buffer(GLenum internalformat, GLuint buffer) {
		glTextureBuffer(_texture, internalformat, buffer);
		return *this;
	}
	NamedTexture &bufferRange(GLenum internalformat, GLuint buffer, GLintptr offset, GLsizei size) {
		glTextureBufferRange(_texture, internalformat, buffer, offset, size);
		return *this;
	}
	NamedTexture &bindUnit(GLuint unit) {
		if (_texture) { glBindTextureUnit(unit, _texture); }
		return *this;
	}
	NamedTexture &bindImageTexture(GLuint unit, GLint level, GLboolean layered, GLint layer,
		GLenum access, GLenum format) {
		glBindImageTexture(unit, _texture, level, layered, layer, access, format);
		return *this;
	}
	NamedTexture &subImage2D(GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
		GLenum format, GLenum type, const void *pixels) {
		glTextureSubImage2D(_texture, level, xoffset, yoffset, width, height, format, type, pixels);
		return *this;
	}
	NamedTexture &parameterf(GLenum pname, GLfloat param) {
		glTextureParameterf(_texture, pname, param);
		return *this;
	}
	GLuint texture() { return _texture; }
	operator GLuint() { return _texture; }
protected:
	TextureTarget _target;
	GLuint _texture = 0;
};

class NamedFramebuffer {	
public:
	NamedFramebuffer &create() {
		if (!_framebuffer) { glCreateFramebuffers(1, &_framebuffer); }
		return *this;
	}
	void destroy() { glDeleteFramebuffers(1, &_framebuffer); _framebuffer = 0; }
	void drawBuffer(GLenum buf) {
		glNamedFramebufferDrawBuffer(_framebuffer, buf);
	}
	void drawBuffers(GLsizei n, const GLenum *bufs) {
		glNamedFramebufferDrawBuffers(_framebuffer, n, bufs);
	}
	void texture2D(GLenum attachment,GLuint texture,GLint level) {
		glNamedFramebufferTexture(_framebuffer, attachment, texture, level);
	}
	void bind(GLenum target) {
		glBindFramebuffer(target, _framebuffer);
	}
	void unbind(GLenum target) {
		glBindFramebuffer(target, 0);
	}
	operator GLuint() { return _framebuffer; }
private:
	GLuint _framebuffer = 0;
};

// -------- -------- -------- -------- -------- -------- -------- --------
// common used simple util.
// 

class Util {
public:
	void init();
	void viewport(int w, int h);
	void viewport(int x, int y, int w, int h);
	void draw_lines(const std::vector<float2> &v, const std::vector<frgba> &c);

private:
	NamedBuffer _vertex_buffer;
	NamedBuffer _color_buffer;

	NamedTexture _vertex_texture;
	NamedTexture _color_texture;

	ShaderProgram _color_program;

	int _vp_x = 0;
	int _vp_y = 0;
	int _vp_w = 1024;
	int _vp_h = 1024;
};

} // end of namespace GLPP

} // end of namespace Mochimazui

#endif