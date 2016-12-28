
#pragma once

namespace Mochimazui {

enum BezierCurveType {
	BCT_Linear = 0x02,
	BCT_Quadratic = 0x03,
	BCT_Cubic = 0x04,
	BCT_Rational = 0x13,
};

enum VGCurveType {
	CT_Linear = BCT_Linear,
	CT_Quadratic = BCT_Quadratic,
	CT_Cubic = BCT_Cubic,
	CT_Rational = BCT_Rational,
};

} // end of namespace Mochimazui
