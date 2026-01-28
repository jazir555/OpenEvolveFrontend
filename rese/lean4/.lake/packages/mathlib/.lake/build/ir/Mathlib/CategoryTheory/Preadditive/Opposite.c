// Lean compiler output
// Module: Mathlib.CategoryTheory.Preadditive.Opposite
// Imports: public import Init public import Mathlib.Algebra.Group.TransferInstance public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Algebra.Module.Opposite public import Mathlib.CategoryTheory.Preadditive.AdditiveFunctor
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_opEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_moduleEndLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CategoryTheory_opHom___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_opHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Preadditive_Opposite_0__CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_moduleEndLeft___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_opHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_addCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Preadditive_Opposite_0__CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_moduleEndLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_End_mulActionLeft___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Preadditive_Opposite_0__CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_CategoryTheory_opEquiv(lean_box(0), x_1, x_3, x_4);
x_6 = lean_apply_2(x_2, x_4, x_3);
x_7 = lp_mathlib_Equiv_addCommGroup___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_instPreadditiveOpposite(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_instPreadditiveOpposite___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_moduleEndLeft(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_End_mulActionLeft___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_moduleEndLeft___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_End_mulActionLeft___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_moduleEndLeft___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_moduleEndLeft(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_unopHom___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_unopHom___lam__0___boxed), 1, 0);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_unopHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_unopHom(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
static lean_object* _init_lp_mathlib_CategoryTheory_opHom___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_unopHom___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_opHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_opHom___closed__0;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_opHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_opHom(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Preadditive_Opposite_0__CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_apply_1(x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Preadditive_Opposite_0__CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_CategoryTheory_Preadditive_Opposite_0__CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_CategoryTheory_Preadditive_Opposite_0__CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___lam__1(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___lam__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___lam__1(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_mathlib_CategoryTheory_opHom___closed__0;
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___lam__1___boxed), 1, 0);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Preadditive_homSelfLinearEquivEndMulOpposite(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CategoryTheory_opHom___closed__0 = _init_lp_mathlib_CategoryTheory_opHom___closed__0();
lean_mark_persistent(lp_mathlib_CategoryTheory_opHom___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
