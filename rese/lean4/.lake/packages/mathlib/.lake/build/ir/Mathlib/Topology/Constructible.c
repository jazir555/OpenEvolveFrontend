// Lean compiler output
// Module: Mathlib.Topology.Constructible
// Imports: public import Init public import Mathlib.Order.BooleanSubalgebra public import Mathlib.Topology.Compactness.Bases public import Mathlib.Topology.LocalAtTarget public import Mathlib.Topology.QuasiSeparated public import Mathlib.Topology.Spectral.Hom public import Mathlib.Topology.Spectral.Prespectral
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BooleanSubalgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Compactness_Bases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_LocalAtTarget(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_QuasiSeparated(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Spectral_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Spectral_Prespectral(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Topology_Constructible(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BooleanSubalgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Compactness_Bases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_LocalAtTarget(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_QuasiSeparated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Spectral_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Spectral_Prespectral(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
