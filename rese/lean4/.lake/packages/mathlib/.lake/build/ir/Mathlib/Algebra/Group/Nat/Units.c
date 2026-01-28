// Lean compiler output
// Module: Mathlib.Algebra.Group.Nat.Units
// Imports: public import Init public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.Group.Units.Defs public import Mathlib.Logic.Unique
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_unique__units;
static lean_object* lp_mathlib_Nat_unique__addUnits___closed__0;
static lean_object* lp_mathlib_Nat_unique__units___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_unique__addUnits;
static lean_object* _init_lp_mathlib_Nat_unique__units___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_unique__units() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_unique__units___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Nat_unique__addUnits___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_unique__addUnits() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Nat_unique__addUnits___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Unique(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_unique__units___closed__0 = _init_lp_mathlib_Nat_unique__units___closed__0();
lean_mark_persistent(lp_mathlib_Nat_unique__units___closed__0);
lp_mathlib_Nat_unique__units = _init_lp_mathlib_Nat_unique__units();
lean_mark_persistent(lp_mathlib_Nat_unique__units);
lp_mathlib_Nat_unique__addUnits___closed__0 = _init_lp_mathlib_Nat_unique__addUnits___closed__0();
lean_mark_persistent(lp_mathlib_Nat_unique__addUnits___closed__0);
lp_mathlib_Nat_unique__addUnits = _init_lp_mathlib_Nat_unique__addUnits();
lean_mark_persistent(lp_mathlib_Nat_unique__addUnits);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
