// Lean compiler output
// Module: Plausible.ArbitraryFueled
// Imports: public import Init public import Plausible.Arbitrary public import Plausible.Gen
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
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled___redArg(lean_object*);
lean_object* lp_plausible_Plausible_Gen_sized(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_sized), 4, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_sized), 4, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Arbitrary(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Gen(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_ArbitraryFueled(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Arbitrary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Gen(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
