// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Subalgebra.Unitization
// Imports: public import Init public import Mathlib.Algebra.Algebra.Unitization public import Mathlib.Algebra.Star.Subalgebra public import Mathlib.GroupTheory.GroupAction.Ring
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
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_unitization___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_unitization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Unitization_starLift___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_unitization___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_unitization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalStarSubalgebra_unitization___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNatAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toIntAlgebra___redArg(lean_object*);
static lean_object* lp_mathlib_NonUnitalSubalgebra_unitization___redArg___closed__0;
lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalStarSubalgebra_unitization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_unitization___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_unitization___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Unitization_lift___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_unitization___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instSemiring;
lean_object* lp_mathlib_NonUnitalStarSubalgebraClass_subtype___lam__0___boxed(lean_object*);
extern lean_object* lp_mathlib_Int_instCommSemiring;
static lean_object* lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg___closed__0;
lean_object* lp_mathlib_SetLike_smul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_unitization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_unitization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_NonUnitalSubalgebra_unitization___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_unitization___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(x_4);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lp_mathlib_SetLike_smul_x27___redArg(x_6);
x_8 = lp_mathlib_Unitization_lift___redArg(x_1, x_5, x_7, x_2, x_3);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_NonUnitalSubalgebra_unitization___redArg___closed__0;
x_11 = lean_apply_1(x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_unitization(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_NonUnitalSubalgebra_unitization___redArg(x_4, x_5, x_6);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_unitization___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_NonUnitalSubalgebra_unitization(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_unitization___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Nat_instSemiring;
x_3 = lp_mathlib_Semiring_toNatAlgebra___redArg(x_1);
x_4 = lp_mathlib_NonUnitalSubalgebra_unitization___redArg(x_2, x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_unitization(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_NonUnitalSubsemiring_unitization___redArg(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_unitization___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_NonUnitalSubsemiring_unitization(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_unitization___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_mathlib_Int_instCommSemiring;
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Ring_toIntAlgebra___redArg(x_1);
x_5 = lp_mathlib_NonUnitalSubalgebra_unitization___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_unitization(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_NonUnitalSubring_unitization___redArg(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_unitization___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_NonUnitalSubring_unitization(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_NonUnitalStarSubalgebraClass_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(x_4);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lp_mathlib_SetLike_smul_x27___redArg(x_6);
x_8 = lp_mathlib_Unitization_starLift___redArg(x_1, x_5, x_7, x_2, x_3);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg___closed__0;
x_11 = lean_apply_1(x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalStarSubalgebra_unitization(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg(x_4, x_6, x_8);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalStarSubalgebra_unitization___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14) {
_start:
{
lean_object* x_15; 
x_15 = lp_mathlib_NonUnitalStarSubalgebra_unitization(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14);
lean_dec(x_14);
lean_dec(x_7);
lean_dec(x_5);
return x_15;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Unitization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Star_Subalgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Unitization(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Unitization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Star_Subalgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_NonUnitalSubalgebra_unitization___redArg___closed__0 = _init_lp_mathlib_NonUnitalSubalgebra_unitization___redArg___closed__0();
lean_mark_persistent(lp_mathlib_NonUnitalSubalgebra_unitization___redArg___closed__0);
lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg___closed__0 = _init_lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg___closed__0();
lean_mark_persistent(lp_mathlib_NonUnitalStarSubalgebra_unitization___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
