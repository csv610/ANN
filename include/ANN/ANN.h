/**
 * @file ANN.h
 * @brief Approximate Nearest Neighbor searching library
 * @author Sunil Arya, David Mount
 * @copyright GNU Lesser General Public License
 */

#ifndef ANN_H
#define ANN_H

#include <ANN/Config.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstring>
#include <string_view>
#include <climits>
#include <cfloat>

namespace ANN {

#ifdef WIN32
  #ifdef DLL_EXPORTS
	 #define DLL_API __declspec(dllexport)
  #else
	#define DLL_API __declspec(dllimport)
  #endif
#else
  #define DLL_API
#endif

//----------------------------------------------------------------------
// Constants and Types
//----------------------------------------------------------------------

const double ANN_DBL_MAX = DBL_MAX;
const double ANN_DIST_INF = DBL_MAX;

#ifdef DBL_DIG
	const int ANNcoordPrec = DBL_DIG;
#else
	const int ANNcoordPrec = 15;
#endif

typedef double ANNcoord;
typedef double ANNdist;
typedef int    ANNidx;

const ANNidx ANN_NULL_IDX = -1;
const bool   ANN_ALLOW_SELF_MATCH = true;

// Legacy compatibility (deprecated)
using ANNbool [[deprecated("Use bool instead")]] = bool;
const bool ANNtrue [[deprecated("Use true instead")]] = true;
const bool ANNfalse [[deprecated("Use false instead")]] = false;
constexpr inline std::string_view ANNversionCmt{""};
constexpr inline std::string_view ANNcopyright{"David M. Mount and Sunil Arya"};
constexpr inline std::string_view ANNlatestRev{"Jan 27, 2010"};

//----------------------------------------------------------------------
// Distance Calculation Utilities
//----------------------------------------------------------------------

[[nodiscard]] constexpr inline ANNdist ANN_POW(ANNcoord v) noexcept { return v * v; }
[[nodiscard]] constexpr inline double  ANN_ROOT(ANNdist x) noexcept { return std::sqrt(x); }
[[nodiscard]] constexpr inline ANNdist ANN_SUM(ANNdist x, ANNdist y) noexcept { return x + y; }
[[nodiscard]] constexpr inline ANNdist ANN_DIFF(ANNcoord x, ANNcoord y) noexcept { return y - x; }

//----------------------------------------------------------------------
// Array types
//----------------------------------------------------------------------

typedef ANNcoord* ANNpoint;         ///< a point
typedef const ANNcoord* ANNpointConst; ///< a constant point
typedef ANNpoint* ANNpointArray;    ///< an array of points 
typedef ANNdist*  ANNdistArray;     ///< an array of distances 
typedef ANNidx*   ANNidxArray;      ///< an array of point indices

//----------------------------------------------------------------------
// Point and Array Utilities
//----------------------------------------------------------------------

[[nodiscard]] DLL_API ANNdist annDist(int dim, ANNpointConst p, ANNpointConst q);

[[nodiscard]] DLL_API ANNpoint annAllocPt(int dim, ANNcoord c = 0);

[[nodiscard]] DLL_API ANNpointArray annAllocPts(int n, int dim);

DLL_API void annDeallocPt(ANNpoint &p);

DLL_API void annDeallocPts(ANNpointArray &pa);

[[nodiscard]] DLL_API ANNpoint annCopyPt(int dim, ANNpointConst source);

//----------------------------------------------------------------------
// ANNpointSet - Abstract Base Class
//----------------------------------------------------------------------

/**
 * @brief Abstract base class for point sets
 */
class DLL_API ANNpointSet {
public:
	virtual ~ANNpointSet() {}

	virtual void annkSearch(
		ANNpointConst   q,
		int				k,
		ANNidxArray		nn_idx,
		ANNdistArray	dd,
		double			eps=0.0
		) = 0;

	virtual int annkFRSearch(
		ANNpointConst   q,
		ANNdist			sqRad,
		int				k = 0,
		ANNidxArray		nn_idx = nullptr,
		ANNdistArray	dd = nullptr,
		double			eps=0.0
		) = 0;

	[[nodiscard]] virtual int theDim() const noexcept = 0;
	[[nodiscard]] virtual int nPoints() const noexcept = 0;
	[[nodiscard]] virtual ANNpointArray thePoints() const noexcept = 0;
};

//----------------------------------------------------------------------
// ANNbruteForce - Brute-force Search
//----------------------------------------------------------------------

class DLL_API ANNbruteForce: public ANNpointSet {
	int				dim;
	int				n_pts;
	ANNpointArray	pts;
public:
	ANNbruteForce(ANNpointArray pa, int n, int dd);
	~ANNbruteForce();

	void annkSearch(
		ANNpointConst   q,
		int				k,
		ANNidxArray		nn_idx,
		ANNdistArray	dd,
		double			eps=0.0) override;

	int annkFRSearch(
		ANNpointConst   q,
		ANNdist			sqRad,
		int				k = 0,
		ANNidxArray		nn_idx = nullptr,
		ANNdistArray	dd = nullptr,
		double			eps=0.0) override;

	[[nodiscard]] int theDim() const noexcept override { return dim; }
	[[nodiscard]] int nPoints() const noexcept override { return n_pts; }
	[[nodiscard]] ANNpointArray thePoints() const noexcept override { return pts; }
};

//----------------------------------------------------------------------
// Search Parameters
//----------------------------------------------------------------------

enum ANNsplitRule {
		ANN_KD_STD				= 0,
		ANN_KD_MIDPT			= 1,
		ANN_KD_FAIR				= 2,
		ANN_KD_SL_MIDPT			= 3,
		ANN_KD_SL_FAIR			= 4,
		ANN_KD_SUGGEST			= 5};
const int ANN_N_SPLIT_RULES		= 6;

enum ANNshrinkRule {
		ANN_BD_NONE				= 0,
		ANN_BD_SIMPLE			= 1,
		ANN_BD_CENTROID			= 2,
		ANN_BD_SUGGEST			= 3};
const int ANN_N_SHRINK_RULES	= 4;

//----------------------------------------------------------------------
// ANNkd_tree - Kd-tree Search Structure
//----------------------------------------------------------------------

class ANNkdStats;
class ANNkd_node;
typedef ANNkd_node*	ANNkd_ptr;

class DLL_API ANNkd_tree: public ANNpointSet {
protected:
	int				dim;
	int				n_pts;
	int				bkt_size;
	ANNpointArray	pts;
	ANNidxArray		pidx;
	ANNkd_ptr		root;
	ANNpoint		bnd_box_lo;
	ANNpoint		bnd_box_hi;

	void SkeletonTree(
		int				n,
		int				dd,
		int				bs,
		ANNpointArray pa = nullptr,
		ANNidxArray pi = nullptr);

public:
	ANNkd_tree(int n = 0, int dd = 0, int bs = 1);

	ANNkd_tree(
		ANNpointArray	pa,
		int				n,
		int				dd,
		int				bs = 1,
		ANNsplitRule	split = ANN_KD_SUGGEST);

	ANNkd_tree(std::istream& in);

	~ANNkd_tree();

	void annkSearch(
		ANNpointConst   q,
		int				k,
		ANNidxArray		nn_idx,
		ANNdistArray	dd,
		double			eps=0.0) override;

	void annkPriSearch( 
		ANNpointConst   q,
		int				k,
		ANNidxArray		nn_idx,
		ANNdistArray	dd,
		double			eps=0.0);

	int annkFRSearch(
		ANNpointConst   q,
		ANNdist			sqRad,
		int				k,
		ANNidxArray		nn_idx = nullptr,
		ANNdistArray	dd = nullptr,
		double			eps=0.0) override;

	[[nodiscard]] int theDim() const noexcept override { return dim; }
	[[nodiscard]] int nPoints() const noexcept override { return n_pts; }
	[[nodiscard]] ANNpointArray thePoints() const noexcept override { return pts; }

	virtual void Print(bool with_pts, std::ostream& out);
	virtual void Dump(bool with_pts, std::ostream& out);
	virtual void getStats(ANNkdStats& st);
};								

//----------------------------------------------------------------------
// ANNbd_tree - Box-decomposition Tree
//----------------------------------------------------------------------

class DLL_API ANNbd_tree: public ANNkd_tree {
public:
	ANNbd_tree(int n, int dd, int bs = 1) : ANNkd_tree(n, dd, bs) {}

	ANNbd_tree(
		ANNpointArray	pa,
		int				n,
		int				dd,
		int				bs = 1,
		ANNsplitRule	split  = ANN_KD_SUGGEST,
		ANNshrinkRule	shrink = ANN_BD_SUGGEST);

	ANNbd_tree(std::istream& in);
};

//----------------------------------------------------------------------
// Global Functions
//----------------------------------------------------------------------

DLL_API void annMaxPtsVisit(int maxPts);
DLL_API void annClose();

} // namespace ANN

#endif
