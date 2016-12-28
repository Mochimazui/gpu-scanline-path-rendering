
#ifndef _MOCHIMAZUI_STRING_UTIL_H_
#define _MOCHIMAZUI_STRING_UTIL_H_

#include <cstdio>
#include <cstring>

#include <set>
#include <string>
#include <sstream>
#include <regex>

namespace Mochimazui {

namespace stdext {

template<class _Elem,
class _Traits,
class _Alloc>
class basic_string : public std::basic_string<_Elem, _Traits, _Alloc> {

	typedef basic_string<_Elem, _Traits, _Alloc> _Myt;
	typedef std::basic_string<_Elem, _Traits, _Alloc> _MyBase;
	//typedef _String_alloc<!is_empty<_Alloc>::value,
	//	_String_base_types<_Elem, _Alloc> > _Mybase;
	typedef _Traits traits_type;
	typedef _Alloc allocator_type;

	//typedef typename _Mybase::_Alty _Alty;

	typedef typename _MyBase::value_type value_type;
	typedef typename _MyBase::size_type size_type;
	typedef typename _MyBase::difference_type difference_type;
	typedef typename _MyBase::pointer pointer;
	typedef typename _MyBase::const_pointer const_pointer;
	typedef typename _MyBase::reference reference;
	typedef typename _MyBase::const_reference const_reference;

	typedef typename _MyBase::iterator iterator;
	typedef typename _MyBase::const_iterator const_iterator;

	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

public:

	basic_string(const _Myt& _Right) :_MyBase(_Right) {}
	basic_string(const _MyBase& _Right) :_MyBase(_Right) {}

	/*
	basic_string(const _Myt& _Right, const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct by copying with allocator
		_Tidy();
		assign(_Right, 0, npos);
	}
	*/

	basic_string() : _MyBase() {}

	/*
	explicit basic_string(const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct empty string with allocator
		_Tidy();
	}

	basic_string(const _Myt& _Right, size_type _Roff,
		size_type _Count = npos)
		: _Mybase(_Right._Getal())
	{	// construct from _Right [_Roff, _Roff + _Count)
		_Tidy();
		assign(_Right, _Roff, _Count);
	}

	basic_string(const _Myt& _Right, size_type _Roff, size_type _Count,
		const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct from _Right [_Roff, _Roff + _Count) with allocator
		_Tidy();
		assign(_Right, _Roff, _Count);
	}

	basic_string(const _Elem *_Ptr, size_type _Count)
		: _Mybase()
	{	// construct from [_Ptr, _Ptr + _Count)
		_Tidy();
		assign(_Ptr, _Count);
	}

	basic_string(const _Elem *_Ptr, size_type _Count, const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct from [_Ptr, _Ptr + _Count) with allocator
		_Tidy();
		assign(_Ptr, _Count);
	}
	*/

	basic_string(const _Elem *_Ptr)
		: _MyBase(_Ptr)
	{	// construct from [_Ptr, <null>)
	}

	/*
	basic_string(const _Elem *_Ptr, const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct from [_Ptr, <null>) with allocator
		_Tidy();
		assign(_Ptr);
	}

	basic_string(size_type _Count, _Elem _Ch)
		: _Mybase()
	{	// construct from _Count * _Ch
		_Tidy();
		assign(_Count, _Ch);
	}

	basic_string(size_type _Count, _Elem _Ch, const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct from _Count * _Ch with allocator
		_Tidy();
		assign(_Count, _Ch);
	}

	template<class _Iter,
	class = typename enable_if<_Is_iterator<_Iter>::value,
		void>::type>
		basic_string(_Iter _First, _Iter _Last)
		: _Mybase()
	{	// construct from [_First, _Last)
		_Tidy();
		_Construct(_First, _Last, _Iter_cat(_First));
	}

	template<class _Iter,
	class = typename enable_if<_Is_iterator<_Iter>::value,
		void>::type>
		basic_string(_Iter _First, _Iter _Last, const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct from [_First, _Last) with allocator
		_Tidy();
		_Construct(_First, _Last, _Iter_cat(_First));
	}

	template<class _Iter>
	void _Construct(_Iter _First,
		_Iter _Last, input_iterator_tag)
	{	// initialize from [_First, _Last), input iterators
		_TRY_BEGIN
			for (; _First != _Last; ++_First)
				append((size_type)1, (_Elem)*_First);
		_CATCH_ALL
			_Tidy(true);
		_RERAISE;
		_CATCH_END
	}

	template<class _Iter>
	void _Construct(_Iter _First,
		_Iter _Last, forward_iterator_tag)
	{	// initialize from [_First, _Last), forward iterators
		_DEBUG_RANGE(_First, _Last);
		size_type _Count = 0;
		_Distance(_First, _Last, _Count);
		reserve(_Count);

		_TRY_BEGIN
			for (; _First != _Last; ++_First)
				append((size_type)1, (_Elem)*_First);
		_CATCH_ALL
			_Tidy(true);
		_RERAISE;
		_CATCH_END
	}

	basic_string(const_pointer _First, const_pointer _Last)
		: _Mybase()
	{	// construct from [_First, _Last), const pointers
		_DEBUG_RANGE(_First, _Last);
		_Tidy();
		if (_First != _Last)
			assign(&*_First, _Last - _First);
	}

	basic_string(const_pointer _First, const_pointer _Last,
		const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct from [_First, _Last), const pointers
		_DEBUG_RANGE(_First, _Last);
		_Tidy();
		if (_First != _Last)
			assign(&*_First, _Last - _First);
	}

	basic_string(const_iterator _First, const_iterator _Last)
		: _Mybase()
	{	// construct from [_First, _Last), const_iterators
		_DEBUG_RANGE(_First, _Last);
		_Tidy();
		if (_First != _Last)
			assign(&*_First, _Last - _First);
	}

	basic_string(_Myt&& _Right) _NOEXCEPT
		: _Mybase(_Right._Getal())
	{	// construct by moving _Right
		_Tidy();
		_Assign_rv(_STD forward<_Myt>(_Right));
	}

	basic_string(_Myt&& _Right, const _Alloc& _Al)
		: _Mybase(_Al)
	{	// construct by moving _Right, allocator
		if (this->_Getal() != _Right._Getal())
			assign(_Right.begin(), _Right.end());
		else
			_Assign_rv(_STD forward<_Myt>(_Right));
	}
	*/

	_Myt substr(size_type _Off = 0, size_type _Count = _MyBase::npos) const
	{	// return [_Off, _Off + _Count) as new string
		//return (_Myt(*this, _Off, _Count, get_allocator()));
		return (_MyBase(*this, _Off, _Count, _MyBase::get_allocator()));
	}

public:

	_Myt left(int n) {
		return substr(0, n);
	}

	_Myt right(int n) {
		int32_t len = (int32_t)_MyBase::length();
		int32_t pos = std::max(len - n, 0);
		return substr(pos);
	}

	int32_t toInt32()  const {
		int value;
		sscanf(_MyBase::c_str(), "%d", &value);
		return value;
	}

	float toFloat() const {
		float v;
		sscanf(_MyBase::c_str(), "%f", &v);
		return v;
	}

	void replace(_Elem a, _Elem b) {
		for (auto i = _MyBase::begin(); i != _MyBase::end(); ++i) {
			if (*i == a) { *i = b; }
		}
	}

	void replace(const _Myt &a, _Elem b) {
		throw "";
	}

	void replace(_Elem a, const _Myt &b) {
		throw "";
	}

	void replace(const _Myt &a, const _Myt &b) {
		throw "";
	}

	std::vector<_Myt> split(_Elem c) const {
		std::vector<_Myt> v;
		_Myt temp;
		for (auto i = _MyBase::begin(); i != _MyBase::end(); ++i) {
			if (*i == c) {
				v.push_back(std::move(temp));
				temp.clear();
			} else {
				temp.push_back(*i);
			}
		}
		if (temp.length()) {
			v.push_back(temp);
		}
		return v;
	}

	std::vector<_Myt> split(const std::set<_Elem> &cs) const {
		std::vector<_Myt> v;
		_Myt temp;
		for (auto i = _MyBase::begin(); i != _MyBase::end(); ++i) {
			if (cs.find(*i) != cs.end() && temp.size()) {
				v.push_back(std::move(temp));
				temp.clear();
			}
			else {
				temp.push_back(*i);
			}
		}
		if (temp.length()) {
			v.push_back(temp);
		}
		return v;
	}

	std::vector<_Myt> split(const _Myt &s) const {
		std::vector<_Myt> v;
		return v;
	}

	std::vector<_Myt> splitLine() {
		std::vector<_Myt> v;
		_Myt temp;
		for (auto i = _MyBase::begin(); i != _MyBase::end(); ++i) {
			switch (*i) {
			default: 
				temp.push_back(*i);
				break;
			case '\r':
				if (((i + 1) != _MyBase::end()) && (*(i + 1) == '\n')) { ++i; }
			case '\n':
				v.push_back(std::move(temp));
				temp.clear();
			}
		}
		if (temp.length()) {
			v.push_back(temp);
		}
		return v;		
	}

};

typedef basic_string<char, std::char_traits<char>, std::allocator<char> >
string;
typedef basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >
wstring;

template<class _Elem, class _Traits, class _Alloc>
inline basic_string<_Elem, _Traits, _Alloc> join(
	const std::vector<basic_string<_Elem, _Traits, _Alloc>>& v) {
	basic_string<_Elem, _Traits, _Alloc> temp;
	for (auto &s : v) {
		temp += s;
	}
	return temp;
}

template<class _Elem, class _Traits, class _Alloc>
inline basic_string<_Elem, _Traits, _Alloc> join(
	const std::vector<basic_string<_Elem, _Traits, _Alloc>>& v, _Elem c) {
	basic_string<_Elem, _Traits, _Alloc> temp;
	for (auto &s : v) {
		if (temp.length()) { temp += c; }
		temp += s;
	}
	return temp;
}

template<class _Elem, class _Traits, class _Alloc>
inline basic_string<_Elem, _Traits, _Alloc> join(
	const std::vector<basic_string<_Elem, _Traits, _Alloc>>& v, 
	const std::vector<basic_string<_Elem, _Traits, _Alloc>>& c) {
	basic_string<_Elem, _Traits, _Alloc> temp;
	for (auto &s : v) {
		if (temp.length()) { temp += c; }
		temp += s;
	}
	return temp;
}

}

/*
int32_t string2int32(const std::string &s) {
	int32_t i;
	sscanf(s.c_str(), "%d", &i);
	return i;
}

uint32_t string2uint32(const std::string &s) {
	uint32_t i;
	sscanf(s.c_str(), "%d", &i);
	return i;
}

float string2float(const std::string &s) {
	float f;
	sscanf(s.c_str(), "%f", &f);
	return f;
}

double string2double(const std::string &s) {
	double d;
	sscanf(s.c_str(), "%lf", &d);
	return d;
}
*/

inline void replaceCommaAndRightBracket(std::string &istr, int i = 0, char r = ' ') {
	for (; i < istr.length(); ++i) {
		switch (istr[i]) {
		case ',':
		case ')':
			istr[i] = r;
			break;
		}
	}
}

inline void replaceCommaAndBracket(std::string &istr, int i = 0, char r = ' ') {
	for (; i < istr.length(); ++i) {
		switch (istr[i]) {
		case ',':
		case '(':
		case ')':
			istr[i] = r;
			break;
		}
	}
}

inline void wspToSpace(std::string &str) {
	for (auto &c : str) {
		switch (c) {
		case ',':
		case '\x20':
		case '\x9':
		case '\xA':
		case '\xD':
			c = ' ';
			break;
		}
	}
}

}

#endif
