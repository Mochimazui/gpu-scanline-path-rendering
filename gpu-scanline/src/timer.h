
#ifndef _SVDAG_CUDA_HEADER_H_
#define _SVDAG_CUDA_HEADER_H_

#pragma warning( push ) 
#pragma warning( disable : 4819 )
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_gl_interop.h>
#pragma warning( pop )

#include <cstdio>
#include <ctime>

#include <chrono>
#include <algorithm>
#include <iostream>

#include <mochimazui/3rd/helper_cuda.h>

namespace Mochimazui {
class Timer {

public:
	void start() {
		_totalTime = std::chrono::system_clock::duration::zero();
		resume();
	}

	void pause() {
		auto end = std::chrono::system_clock::now();
		_totalTime += end - _start;
	}

	void resume() {
		_start = std::chrono::system_clock::now();
	}

	void end() {
		pause();
	}

public:

	void start(const std::string &msg) {
		_msg = msg;
		printf("%s START.\n", msg.c_str());
		start();
	}

	void pause(const std::string &msg) {
		pause();
		printf("%s PAUSE.\n", msg.c_str());
	}

	void resume(const std::string &msg) {
		resume();
		printf("%s RESUME.\n", msg.c_str());
	}

	void end(const std::string& msg) {
		end();
		auto omsg = msg == "" ? _msg : msg;
		std::cout << omsg << " END" << std::endl;
		std::cout << omsg << " duration = "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(_totalTime).count()
			<< "ms." << std::endl;
	}

public:
	std::chrono::system_clock::duration time() {
		return _totalTime;
	}

	float time_in_ms() {
		return std::chrono::duration_cast<std::chrono::microseconds>(_totalTime).count() / 1000.f;
	}

private:

	std::string _msg;
	std::chrono::system_clock::time_point _start;
	std::chrono::system_clock::duration _totalTime;
};


class CUDATimer {

public:
	CUDATimer() {
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
	}

	void start() {
		cudaEventRecord(_start);
	}

	void start(const std::string &msg) {
		_msg = msg;
		start();
	}

	float stop() {
		cudaEventRecord(_stop);
		cudaEventSynchronize(_stop);
		_ms = 0;
		cudaEventElapsedTime(&_ms, _start, _stop);
		return _ms;
	}

	float stop(const std::string &msg) {
		stop();
		if (msg.length() != 0) { _msg = msg; }
		std::cout << _msg << " END." << std::endl;
		std::cout << _msg << " duration = " << _ms << " ms." << std::endl;
		return _ms;
	}

private:
	std::string _msg;
	cudaEvent_t _start, _stop;
	float _ms;
};
}

static Mochimazui::CUDATimer g_timer_zero;
static int g_timer_enabled = 0;
static inline void timer_reset() {
	//clock_gettime(CLOCK_MONOTONIC,&g_timer_zero);
	g_timer_zero.start();
	g_timer_enabled = 1;
}
static inline void timer_print(const char* msg, const char* file, int line) {
	if (!g_timer_enabled) { return; }
	//timespec cur_time;
	cudaDeviceSynchronize();
	__getLastCudaError(msg, file, line);
	//clock_gettime(CLOCK_MONOTONIC,&cur_time);
	//double dt=(double)(cur_time.tv_sec-g_timer_zero.tv_sec)+(double)(cur_time.tv_nsec-g_timer_zero.tv_nsec)*1e-9;
	float dt = g_timer_zero.stop();
	//g_timer_zero=cur_time;
	printf(">>> %s %5.2f ms\n", msg, dt);
	g_timer_zero.start();
}
static inline void timer_done() {
	g_timer_enabled = 0;
}

#endif