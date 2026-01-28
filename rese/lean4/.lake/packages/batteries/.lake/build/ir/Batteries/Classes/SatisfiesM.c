// Lean compiler output
// Module: Batteries.Classes.SatisfiesM
// Imports: public import Init public import Batteries.Lean.EStateM public import Batteries.Lean.Except
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
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__ExceptT_bindCont_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instId;
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExcept___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__ExceptT__eq_match__1__11_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateRefT_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instId___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Except_pmap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__ExceptT_bindCont_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instEStateM(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instEStateM___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateRefT_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExcept(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__EStateM__eq_match__1__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateRefT_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__MonadSatisfying_instEStateM_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instOption;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__MonadSatisfying_instEStateM_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instId___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__ExceptT__eq_match__1__11_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__EStateM__eq_match__1__3_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__EStateM__eq_match__1__3_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_dec(x_3);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_3(x_2, x_4, x_5, lean_box(0));
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec(x_2);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_apply_3(x_3, x_7, x_8, lean_box(0));
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__EStateM__eq_match__1__3_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__EStateM__eq_match__1__3_splitter___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__ExceptT__eq_match__1__11_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_2);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_3, x_4, lean_box(0));
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_2(x_2, x_6, lean_box(0));
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__ExceptT__eq_match__1__11_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries___private_Batteries_Classes_SatisfiesM_0__SatisfiesM__ExceptT__eq_match__1__11_splitter___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__ExceptT_bindCont_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_2);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_3, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_2, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__ExceptT_bindCont_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries___private_Batteries_Classes_SatisfiesM_0__ExceptT_bindCont_match__1_splitter___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instId___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instId___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_MonadSatisfying_instId___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
static lean_object* _init_lp_batteries_MonadSatisfying_instId() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instId___lam__0___boxed), 4, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instOption___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_dec(x_3);
x_8 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
}
}
static lean_object* _init_lp_batteries_MonadSatisfying_instOption() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instOption___lam__0), 4, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExcept___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
return x_3;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec(x_3);
x_7 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
else
{
uint8_t x_8; 
x_8 = !lean_is_exclusive(x_3);
if (x_8 == 0)
{
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
lean_dec(x_3);
x_10 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_10, 0, x_9);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExcept(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instExcept___lam__0), 4, 0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_apply_1(x_4, x_6);
x_8 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_7, lean_box(0));
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instReaderT___redArg___lam__0), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_MonadSatisfying_instReaderT___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instReaderT___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_MonadSatisfying_instReaderT(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateRefT_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instReaderT___redArg___lam__0), 6, 1);
lean_closure_set(x_7, 0, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateRefT_x27___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instReaderT___redArg___lam__0), 6, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateRefT_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_MonadSatisfying_instStateRefT_x27(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT___redArg___lam__0(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_inc(x_3);
lean_dec(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_apply_1(x_6, x_8);
x_11 = lean_apply_4(x_2, lean_box(0), lean_box(0), x_10, lean_box(0));
x_12 = lean_apply_4(x_9, lean_box(0), lean_box(0), x_3, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instStateT___redArg___lam__0), 1, 0);
x_6 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instStateT___redArg___lam__1), 8, 3);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_2);
lean_closure_set(x_6, 2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instStateT(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_MonadSatisfying_instStateT___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_MonadSatisfying_instExceptT___redArg___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Except_pmap___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_apply_4(x_2, lean_box(0), lean_box(0), x_6, lean_box(0));
x_10 = lean_apply_4(x_8, lean_box(0), lean_box(0), x_3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instExceptT___redArg___lam__0___boxed), 2, 0);
x_6 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instExceptT___redArg___lam__1), 2, 1);
lean_closure_set(x_6, 0, x_5);
x_7 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instExceptT___redArg___lam__2), 7, 3);
lean_closure_set(x_7, 0, x_4);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instExceptT(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_batteries_MonadSatisfying_instExceptT___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__MonadSatisfying_instEStateM_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_dec(x_3);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_3(x_2, x_4, x_5, lean_box(0));
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec(x_2);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_apply_3(x_3, x_7, x_8, lean_box(0));
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Classes_SatisfiesM_0__MonadSatisfying_instEStateM_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries___private_Batteries_Classes_SatisfiesM_0__MonadSatisfying_instEStateM_match__1_splitter___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instEStateM___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_apply_1(x_3, x_5);
if (lean_obj_tag(x_6) == 0)
{
uint8_t x_7; 
x_7 = !lean_is_exclusive(x_6);
if (x_7 == 0)
{
return x_6;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_6, 0);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_6);
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
else
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_6);
if (x_11 == 0)
{
return x_6;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_6, 0);
x_13 = lean_ctor_get(x_6, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_6);
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MonadSatisfying_instEStateM(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_batteries_MonadSatisfying_instEStateM___lam__0), 5, 0);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_EStateM(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Except(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Classes_SatisfiesM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_EStateM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Except(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_MonadSatisfying_instId = _init_lp_batteries_MonadSatisfying_instId();
lean_mark_persistent(lp_batteries_MonadSatisfying_instId);
lp_batteries_MonadSatisfying_instOption = _init_lp_batteries_MonadSatisfying_instOption();
lean_mark_persistent(lp_batteries_MonadSatisfying_instOption);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
