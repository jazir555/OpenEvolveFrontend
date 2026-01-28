// Lean compiler output
// Module: Mathlib.Algebra.CharP.Invertible
// Imports: public import Init public import Mathlib.Algebra.CharP.Defs public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Ring.Parity public import Mathlib.Algebra.GroupWithZero.Invertible public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Data.Int.GCD public import Mathlib.Data.Nat.Cast.Commute
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
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPos___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleTwo(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRingCharNotDvd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleTwo___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCoprime___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCoprime(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCharPNotDvd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleThree___redArg(lean_object*);
lean_object* lp_mathlib_invertibleOfNonzero___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_gcdA(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCharPNotDvd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCharPNotDvd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleThree(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRingCharNotDvd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCoprime___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_Nat_gcdA(x_3, x_2);
x_7 = lean_apply_1(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCoprime(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_invertibleOfCoprime___redArg(x_2, x_3, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRingCharNotDvd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_1);
lean_inc_ref(x_3);
x_4 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_3);
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_8, x_2);
x_10 = lp_mathlib_invertibleOfNonzero___redArg(x_4, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfRingCharNotDvd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_invertibleOfRingCharNotDvd___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCharPNotDvd___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_1);
lean_inc_ref(x_3);
x_4 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_3);
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_8, x_2);
x_10 = lp_mathlib_invertibleOfNonzero___redArg(x_4, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCharPNotDvd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_invertibleOfCharPNotDvd___redArg(x_2, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfCharPNotDvd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_invertibleOfCharPNotDvd(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPos___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_3 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_1);
lean_inc_ref(x_3);
x_4 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_3);
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_5);
lean_dec_ref(x_5);
x_7 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_6);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lean_apply_1(x_8, x_2);
x_10 = lp_mathlib_invertibleOfNonzero___redArg(x_4, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPos(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_invertibleOfPos___redArg(x_2, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_1);
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_4);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_unsigned_to_nat(1u);
x_9 = lean_nat_add(x_2, x_8);
x_10 = lean_apply_1(x_7, x_9);
x_11 = lp_mathlib_invertibleOfNonzero___redArg(x_3, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_invertibleSucc___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_invertibleSucc(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleSucc___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_invertibleSucc___redArg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleTwo___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_1);
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
lean_dec_ref(x_3);
x_5 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_unsigned_to_nat(2u);
x_8 = lean_apply_1(x_6, x_7);
x_9 = lp_mathlib_invertibleOfNonzero___redArg(x_2, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleTwo(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_invertibleTwo___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleThree___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(x_1);
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_3);
lean_dec_ref(x_3);
x_5 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_unsigned_to_nat(3u);
x_8 = lean_apply_1(x_6, x_7);
x_9 = lp_mathlib_invertibleOfNonzero___redArg(x_2, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleThree(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_invertibleThree___redArg(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Parity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_GCD(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Commute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Invertible(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharP_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_GCD(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Commute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
