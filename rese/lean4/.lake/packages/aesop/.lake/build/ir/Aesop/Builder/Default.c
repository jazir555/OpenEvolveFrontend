// Lean compiler output
// Module: Aesop.Builder.Default
// Imports: public import Init public import Aesop.Builder.Constructors public import Aesop.Builder.NormSimp public import Aesop.Builder.Tactic public import Aesop.Builder.Apply
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_RuleBuilder_default___closed__2;
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_constructors___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1;
uint8_t lp_aesop_Aesop_PhaseSpec_phase(lean_object*);
static lean_object* lp_aesop_Aesop_RuleBuilder_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5;
static lean_object* lp_aesop_Aesop_RuleBuilder_default___closed__0;
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_tactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_7 = lean_st_ref_get(x_5);
x_8 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_8);
lean_dec(x_7);
x_9 = lean_st_ref_get(x_3);
x_10 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_10);
lean_dec(x_9);
x_11 = lean_ctor_get(x_2, 2);
x_12 = lean_ctor_get(x_4, 2);
lean_inc(x_12);
lean_inc_ref(x_11);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_8);
lean_ctor_set(x_13, 1, x_10);
lean_ctor_set(x_13, 2, x_11);
lean_ctor_set(x_13, 3, x_12);
x_14 = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_1);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_7 = lean_ctor_get(x_4, 5);
x_8 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_7);
lean_ctor_set(x_11, 1, x_10);
lean_ctor_set_tag(x_8, 1);
lean_ctor_set(x_8, 0, x_11);
return x_8;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_8, 0);
lean_inc(x_12);
lean_dec(x_8);
lean_inc(x_7);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_7);
lean_ctor_set(x_13, 1, x_12);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(x_2, x_6, x_7, x_8, x_9);
return x_11;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("aesop: Unable to interpret '", 28, 28);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("' as ", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" rule. Try specifying a builder.", 32, 32);
return x_1;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4;
x_2 = l_Lean_stringToMessageData(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_11 = lean_ctor_get(x_2, 0);
lean_inc(x_11);
lean_dec_ref(x_2);
x_12 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1;
x_13 = l_Lean_MessageData_ofSyntax(x_11);
x_14 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3;
x_16 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
x_17 = l_Lean_stringToMessageData(x_1);
x_18 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
x_19 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5;
x_20 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set(x_20, 1, x_19);
x_21 = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(x_20, x_6, x_7, x_8, x_9);
return x_21;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_7;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_default___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("an unsafe", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_default___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("a norm", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_default___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("a safe", 6, 6);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_24; lean_object* x_39; lean_object* x_40; lean_object* x_41; uint8_t x_42; lean_object* x_50; lean_object* x_51; lean_object* x_52; uint8_t x_53; lean_object* x_68; lean_object* x_69; lean_object* x_70; uint8_t x_71; lean_object* x_86; lean_object* x_87; lean_object* x_88; uint8_t x_89; lean_object* x_97; lean_object* x_98; lean_object* x_99; uint8_t x_100; lean_object* x_115; uint8_t x_116; 
x_115 = lean_ctor_get(x_1, 2);
x_116 = lp_aesop_Aesop_PhaseSpec_phase(x_115);
switch (x_116) {
case 0:
{
lean_object* x_117; 
x_117 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_117) == 0)
{
lean_object* x_118; lean_object* x_119; 
x_118 = lean_ctor_get(x_117, 0);
lean_inc(x_118);
lean_dec_ref(x_117);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_119 = lp_aesop_Aesop_RuleBuilder_constructors___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_119) == 0)
{
lean_dec(x_118);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_119;
}
else
{
lean_object* x_120; uint8_t x_121; uint8_t x_136; 
x_120 = lean_ctor_get(x_119, 0);
lean_inc(x_120);
x_136 = l_Lean_Exception_isInterrupt(x_120);
if (x_136 == 0)
{
uint8_t x_137; 
x_137 = l_Lean_Exception_isRuntime(x_120);
x_121 = x_137;
goto block_135;
}
else
{
lean_dec(x_120);
x_121 = x_136;
goto block_135;
}
block_135:
{
if (x_121 == 0)
{
lean_object* x_122; 
lean_dec_ref(x_119);
x_122 = l_Lean_Meta_SavedState_restore___redArg(x_118, x_6, x_8);
lean_dec(x_118);
if (lean_obj_tag(x_122) == 0)
{
lean_object* x_123; 
lean_dec_ref(x_122);
x_123 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_123) == 0)
{
lean_object* x_124; lean_object* x_125; 
x_124 = lean_ctor_get(x_123, 0);
lean_inc(x_124);
lean_dec_ref(x_123);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_125 = lp_aesop_Aesop_RuleBuilder_tactic(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_125) == 0)
{
lean_dec(x_124);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_125;
}
else
{
lean_object* x_126; uint8_t x_127; 
x_126 = lean_ctor_get(x_125, 0);
lean_inc(x_126);
x_127 = l_Lean_Exception_isInterrupt(x_126);
if (x_127 == 0)
{
uint8_t x_128; 
x_128 = l_Lean_Exception_isRuntime(x_126);
x_68 = x_124;
x_69 = lean_box(0);
x_70 = x_125;
x_71 = x_128;
goto block_85;
}
else
{
lean_dec(x_126);
x_68 = x_124;
x_69 = lean_box(0);
x_70 = x_125;
x_71 = x_127;
goto block_85;
}
}
}
else
{
uint8_t x_129; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_129 = !lean_is_exclusive(x_123);
if (x_129 == 0)
{
return x_123;
}
else
{
lean_object* x_130; lean_object* x_131; 
x_130 = lean_ctor_get(x_123, 0);
lean_inc(x_130);
lean_dec(x_123);
x_131 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_131, 0, x_130);
return x_131;
}
}
}
else
{
uint8_t x_132; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_132 = !lean_is_exclusive(x_122);
if (x_132 == 0)
{
return x_122;
}
else
{
lean_object* x_133; lean_object* x_134; 
x_133 = lean_ctor_get(x_122, 0);
lean_inc(x_133);
lean_dec(x_122);
x_134 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_134, 0, x_133);
return x_134;
}
}
}
else
{
lean_dec(x_118);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_119;
}
}
}
}
else
{
uint8_t x_138; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_138 = !lean_is_exclusive(x_117);
if (x_138 == 0)
{
return x_117;
}
else
{
lean_object* x_139; lean_object* x_140; 
x_139 = lean_ctor_get(x_117, 0);
lean_inc(x_139);
lean_dec(x_117);
x_140 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_140, 0, x_139);
return x_140;
}
}
}
case 1:
{
lean_object* x_141; 
x_141 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_141) == 0)
{
lean_object* x_142; lean_object* x_143; 
x_142 = lean_ctor_get(x_141, 0);
lean_inc(x_142);
lean_dec_ref(x_141);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_143 = lp_aesop_Aesop_RuleBuilder_constructors___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_143) == 0)
{
lean_dec(x_142);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_143;
}
else
{
lean_object* x_144; uint8_t x_145; uint8_t x_160; 
x_144 = lean_ctor_get(x_143, 0);
lean_inc(x_144);
x_160 = l_Lean_Exception_isInterrupt(x_144);
if (x_160 == 0)
{
uint8_t x_161; 
x_161 = l_Lean_Exception_isRuntime(x_144);
x_145 = x_161;
goto block_159;
}
else
{
lean_dec(x_144);
x_145 = x_160;
goto block_159;
}
block_159:
{
if (x_145 == 0)
{
lean_object* x_146; 
lean_dec_ref(x_143);
x_146 = l_Lean_Meta_SavedState_restore___redArg(x_142, x_6, x_8);
lean_dec(x_142);
if (lean_obj_tag(x_146) == 0)
{
lean_object* x_147; 
lean_dec_ref(x_146);
x_147 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_147) == 0)
{
lean_object* x_148; lean_object* x_149; 
x_148 = lean_ctor_get(x_147, 0);
lean_inc(x_148);
lean_dec_ref(x_147);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_149 = lp_aesop_Aesop_RuleBuilder_tactic(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_149) == 0)
{
lean_dec(x_148);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_149;
}
else
{
lean_object* x_150; uint8_t x_151; 
x_150 = lean_ctor_get(x_149, 0);
lean_inc(x_150);
x_151 = l_Lean_Exception_isInterrupt(x_150);
if (x_151 == 0)
{
uint8_t x_152; 
x_152 = l_Lean_Exception_isRuntime(x_150);
x_97 = lean_box(0);
x_98 = x_148;
x_99 = x_149;
x_100 = x_152;
goto block_114;
}
else
{
lean_dec(x_150);
x_97 = lean_box(0);
x_98 = x_148;
x_99 = x_149;
x_100 = x_151;
goto block_114;
}
}
}
else
{
uint8_t x_153; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_153 = !lean_is_exclusive(x_147);
if (x_153 == 0)
{
return x_147;
}
else
{
lean_object* x_154; lean_object* x_155; 
x_154 = lean_ctor_get(x_147, 0);
lean_inc(x_154);
lean_dec(x_147);
x_155 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_155, 0, x_154);
return x_155;
}
}
}
else
{
uint8_t x_156; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_156 = !lean_is_exclusive(x_146);
if (x_156 == 0)
{
return x_146;
}
else
{
lean_object* x_157; lean_object* x_158; 
x_157 = lean_ctor_get(x_146, 0);
lean_inc(x_157);
lean_dec(x_146);
x_158 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_158, 0, x_157);
return x_158;
}
}
}
else
{
lean_dec(x_142);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_143;
}
}
}
}
else
{
uint8_t x_162; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_162 = !lean_is_exclusive(x_141);
if (x_162 == 0)
{
return x_141;
}
else
{
lean_object* x_163; lean_object* x_164; 
x_163 = lean_ctor_get(x_141, 0);
lean_inc(x_163);
lean_dec(x_141);
x_164 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_164, 0, x_163);
return x_164;
}
}
}
default: 
{
lean_object* x_165; 
x_165 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_165) == 0)
{
lean_object* x_166; lean_object* x_167; 
x_166 = lean_ctor_get(x_165, 0);
lean_inc(x_166);
lean_dec_ref(x_165);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_167 = lp_aesop_Aesop_RuleBuilder_constructors___redArg(x_1, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_167) == 0)
{
lean_dec(x_166);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_167;
}
else
{
lean_object* x_168; uint8_t x_169; uint8_t x_184; 
x_168 = lean_ctor_get(x_167, 0);
lean_inc(x_168);
x_184 = l_Lean_Exception_isInterrupt(x_168);
if (x_184 == 0)
{
uint8_t x_185; 
x_185 = l_Lean_Exception_isRuntime(x_168);
x_169 = x_185;
goto block_183;
}
else
{
lean_dec(x_168);
x_169 = x_184;
goto block_183;
}
block_183:
{
if (x_169 == 0)
{
lean_object* x_170; 
lean_dec_ref(x_167);
x_170 = l_Lean_Meta_SavedState_restore___redArg(x_166, x_6, x_8);
lean_dec(x_166);
if (lean_obj_tag(x_170) == 0)
{
lean_object* x_171; 
lean_dec_ref(x_170);
x_171 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_171) == 0)
{
lean_object* x_172; lean_object* x_173; 
x_172 = lean_ctor_get(x_171, 0);
lean_inc(x_172);
lean_dec_ref(x_171);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_173 = lp_aesop_Aesop_RuleBuilder_tactic(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_173) == 0)
{
lean_dec(x_172);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_173;
}
else
{
lean_object* x_174; uint8_t x_175; 
x_174 = lean_ctor_get(x_173, 0);
lean_inc(x_174);
x_175 = l_Lean_Exception_isInterrupt(x_174);
if (x_175 == 0)
{
uint8_t x_176; 
x_176 = l_Lean_Exception_isRuntime(x_174);
x_21 = lean_box(0);
x_22 = x_173;
x_23 = x_172;
x_24 = x_176;
goto block_38;
}
else
{
lean_dec(x_174);
x_21 = lean_box(0);
x_22 = x_173;
x_23 = x_172;
x_24 = x_175;
goto block_38;
}
}
}
else
{
uint8_t x_177; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_177 = !lean_is_exclusive(x_171);
if (x_177 == 0)
{
return x_171;
}
else
{
lean_object* x_178; lean_object* x_179; 
x_178 = lean_ctor_get(x_171, 0);
lean_inc(x_178);
lean_dec(x_171);
x_179 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_179, 0, x_178);
return x_179;
}
}
}
else
{
uint8_t x_180; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_180 = !lean_is_exclusive(x_170);
if (x_180 == 0)
{
return x_170;
}
else
{
lean_object* x_181; lean_object* x_182; 
x_181 = lean_ctor_get(x_170, 0);
lean_inc(x_181);
lean_dec(x_170);
x_182 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_182, 0, x_181);
return x_182;
}
}
}
else
{
lean_dec(x_166);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_167;
}
}
}
}
else
{
uint8_t x_186; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_186 = !lean_is_exclusive(x_165);
if (x_186 == 0)
{
return x_165;
}
else
{
lean_object* x_187; lean_object* x_188; 
x_187 = lean_ctor_get(x_165, 0);
lean_inc(x_187);
lean_dec(x_165);
x_188 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_188, 0, x_187);
return x_188;
}
}
}
}
block_20:
{
if (x_13 == 0)
{
lean_object* x_14; 
lean_dec_ref(x_10);
x_14 = l_Lean_Meta_SavedState_restore___redArg(x_12, x_6, x_8);
lean_dec_ref(x_12);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; 
lean_dec_ref(x_14);
x_15 = lp_aesop_Aesop_RuleBuilder_default___closed__0;
x_16 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(x_15, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_16;
}
else
{
uint8_t x_17; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_17 = !lean_is_exclusive(x_14);
if (x_17 == 0)
{
return x_14;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = lean_ctor_get(x_14, 0);
lean_inc(x_18);
lean_dec(x_14);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
else
{
lean_dec_ref(x_12);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
block_38:
{
if (x_24 == 0)
{
lean_object* x_25; 
lean_dec_ref(x_22);
x_25 = l_Lean_Meta_SavedState_restore___redArg(x_23, x_6, x_8);
lean_dec_ref(x_23);
if (lean_obj_tag(x_25) == 0)
{
lean_object* x_26; 
lean_dec_ref(x_25);
x_26 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; lean_object* x_28; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec_ref(x_26);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_28 = lp_aesop_Aesop_RuleBuilder_apply(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_28) == 0)
{
lean_dec(x_27);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_28;
}
else
{
lean_object* x_29; uint8_t x_30; 
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
x_30 = l_Lean_Exception_isInterrupt(x_29);
if (x_30 == 0)
{
uint8_t x_31; 
x_31 = l_Lean_Exception_isRuntime(x_29);
x_10 = x_28;
x_11 = lean_box(0);
x_12 = x_27;
x_13 = x_31;
goto block_20;
}
else
{
lean_dec(x_29);
x_10 = x_28;
x_11 = lean_box(0);
x_12 = x_27;
x_13 = x_30;
goto block_20;
}
}
}
else
{
uint8_t x_32; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_32 = !lean_is_exclusive(x_26);
if (x_32 == 0)
{
return x_26;
}
else
{
lean_object* x_33; lean_object* x_34; 
x_33 = lean_ctor_get(x_26, 0);
lean_inc(x_33);
lean_dec(x_26);
x_34 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_34, 0, x_33);
return x_34;
}
}
}
else
{
uint8_t x_35; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_35 = !lean_is_exclusive(x_25);
if (x_35 == 0)
{
return x_25;
}
else
{
lean_object* x_36; lean_object* x_37; 
x_36 = lean_ctor_get(x_25, 0);
lean_inc(x_36);
lean_dec(x_25);
x_37 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_37, 0, x_36);
return x_37;
}
}
}
else
{
lean_dec_ref(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_22;
}
}
block_49:
{
if (x_42 == 0)
{
lean_object* x_43; 
lean_dec_ref(x_41);
x_43 = l_Lean_Meta_SavedState_restore___redArg(x_40, x_6, x_8);
lean_dec_ref(x_40);
if (lean_obj_tag(x_43) == 0)
{
lean_object* x_44; lean_object* x_45; 
lean_dec_ref(x_43);
x_44 = lp_aesop_Aesop_RuleBuilder_default___closed__1;
x_45 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(x_44, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_45;
}
else
{
uint8_t x_46; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_46 = !lean_is_exclusive(x_43);
if (x_46 == 0)
{
return x_43;
}
else
{
lean_object* x_47; lean_object* x_48; 
x_47 = lean_ctor_get(x_43, 0);
lean_inc(x_47);
lean_dec(x_43);
x_48 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_48, 0, x_47);
return x_48;
}
}
}
else
{
lean_dec_ref(x_40);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_41;
}
}
block_67:
{
if (x_53 == 0)
{
lean_object* x_54; 
lean_dec_ref(x_51);
x_54 = l_Lean_Meta_SavedState_restore___redArg(x_52, x_6, x_8);
lean_dec_ref(x_52);
if (lean_obj_tag(x_54) == 0)
{
lean_object* x_55; 
lean_dec_ref(x_54);
x_55 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_55) == 0)
{
lean_object* x_56; lean_object* x_57; 
x_56 = lean_ctor_get(x_55, 0);
lean_inc(x_56);
lean_dec_ref(x_55);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_57 = lp_aesop_Aesop_RuleBuilder_apply(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_57) == 0)
{
lean_dec(x_56);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_57;
}
else
{
lean_object* x_58; uint8_t x_59; 
x_58 = lean_ctor_get(x_57, 0);
lean_inc(x_58);
x_59 = l_Lean_Exception_isInterrupt(x_58);
if (x_59 == 0)
{
uint8_t x_60; 
x_60 = l_Lean_Exception_isRuntime(x_58);
x_39 = lean_box(0);
x_40 = x_56;
x_41 = x_57;
x_42 = x_60;
goto block_49;
}
else
{
lean_dec(x_58);
x_39 = lean_box(0);
x_40 = x_56;
x_41 = x_57;
x_42 = x_59;
goto block_49;
}
}
}
else
{
uint8_t x_61; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_61 = !lean_is_exclusive(x_55);
if (x_61 == 0)
{
return x_55;
}
else
{
lean_object* x_62; lean_object* x_63; 
x_62 = lean_ctor_get(x_55, 0);
lean_inc(x_62);
lean_dec(x_55);
x_63 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_63, 0, x_62);
return x_63;
}
}
}
else
{
uint8_t x_64; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_64 = !lean_is_exclusive(x_54);
if (x_64 == 0)
{
return x_54;
}
else
{
lean_object* x_65; lean_object* x_66; 
x_65 = lean_ctor_get(x_54, 0);
lean_inc(x_65);
lean_dec(x_54);
x_66 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_66, 0, x_65);
return x_66;
}
}
}
else
{
lean_dec_ref(x_52);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_51;
}
}
block_85:
{
if (x_71 == 0)
{
lean_object* x_72; 
lean_dec_ref(x_70);
x_72 = l_Lean_Meta_SavedState_restore___redArg(x_68, x_6, x_8);
lean_dec_ref(x_68);
if (lean_obj_tag(x_72) == 0)
{
lean_object* x_73; 
lean_dec_ref(x_72);
x_73 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_73) == 0)
{
lean_object* x_74; lean_object* x_75; 
x_74 = lean_ctor_get(x_73, 0);
lean_inc(x_74);
lean_dec_ref(x_73);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_75 = lp_aesop_Aesop_RuleBuilder_simp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_75) == 0)
{
lean_dec(x_74);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_75;
}
else
{
lean_object* x_76; uint8_t x_77; 
x_76 = lean_ctor_get(x_75, 0);
lean_inc(x_76);
x_77 = l_Lean_Exception_isInterrupt(x_76);
if (x_77 == 0)
{
uint8_t x_78; 
x_78 = l_Lean_Exception_isRuntime(x_76);
x_50 = lean_box(0);
x_51 = x_75;
x_52 = x_74;
x_53 = x_78;
goto block_67;
}
else
{
lean_dec(x_76);
x_50 = lean_box(0);
x_51 = x_75;
x_52 = x_74;
x_53 = x_77;
goto block_67;
}
}
}
else
{
uint8_t x_79; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_79 = !lean_is_exclusive(x_73);
if (x_79 == 0)
{
return x_73;
}
else
{
lean_object* x_80; lean_object* x_81; 
x_80 = lean_ctor_get(x_73, 0);
lean_inc(x_80);
lean_dec(x_73);
x_81 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_81, 0, x_80);
return x_81;
}
}
}
else
{
uint8_t x_82; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_82 = !lean_is_exclusive(x_72);
if (x_82 == 0)
{
return x_72;
}
else
{
lean_object* x_83; lean_object* x_84; 
x_83 = lean_ctor_get(x_72, 0);
lean_inc(x_83);
lean_dec(x_72);
x_84 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_84, 0, x_83);
return x_84;
}
}
}
else
{
lean_dec_ref(x_68);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_70;
}
}
block_96:
{
if (x_89 == 0)
{
lean_object* x_90; 
lean_dec_ref(x_88);
x_90 = l_Lean_Meta_SavedState_restore___redArg(x_87, x_6, x_8);
lean_dec_ref(x_87);
if (lean_obj_tag(x_90) == 0)
{
lean_object* x_91; lean_object* x_92; 
lean_dec_ref(x_90);
x_91 = lp_aesop_Aesop_RuleBuilder_default___closed__2;
x_92 = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(x_91, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_92;
}
else
{
uint8_t x_93; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_93 = !lean_is_exclusive(x_90);
if (x_93 == 0)
{
return x_90;
}
else
{
lean_object* x_94; lean_object* x_95; 
x_94 = lean_ctor_get(x_90, 0);
lean_inc(x_94);
lean_dec(x_90);
x_95 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_95, 0, x_94);
return x_95;
}
}
}
else
{
lean_dec_ref(x_87);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_88;
}
}
block_114:
{
if (x_100 == 0)
{
lean_object* x_101; 
lean_dec_ref(x_99);
x_101 = l_Lean_Meta_SavedState_restore___redArg(x_98, x_6, x_8);
lean_dec_ref(x_98);
if (lean_obj_tag(x_101) == 0)
{
lean_object* x_102; 
lean_dec_ref(x_101);
x_102 = l_Lean_Meta_saveState___redArg(x_6, x_8);
if (lean_obj_tag(x_102) == 0)
{
lean_object* x_103; lean_object* x_104; 
x_103 = lean_ctor_get(x_102, 0);
lean_inc(x_103);
lean_dec_ref(x_102);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc(x_4);
lean_inc_ref(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_104 = lp_aesop_Aesop_RuleBuilder_apply(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_104) == 0)
{
lean_dec(x_103);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_104;
}
else
{
lean_object* x_105; uint8_t x_106; 
x_105 = lean_ctor_get(x_104, 0);
lean_inc(x_105);
x_106 = l_Lean_Exception_isInterrupt(x_105);
if (x_106 == 0)
{
uint8_t x_107; 
x_107 = l_Lean_Exception_isRuntime(x_105);
x_86 = lean_box(0);
x_87 = x_103;
x_88 = x_104;
x_89 = x_107;
goto block_96;
}
else
{
lean_dec(x_105);
x_86 = lean_box(0);
x_87 = x_103;
x_88 = x_104;
x_89 = x_106;
goto block_96;
}
}
}
else
{
uint8_t x_108; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_108 = !lean_is_exclusive(x_102);
if (x_108 == 0)
{
return x_102;
}
else
{
lean_object* x_109; lean_object* x_110; 
x_109 = lean_ctor_get(x_102, 0);
lean_inc(x_109);
lean_dec(x_102);
x_110 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_110, 0, x_109);
return x_110;
}
}
}
else
{
uint8_t x_111; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_111 = !lean_is_exclusive(x_101);
if (x_111 == 0)
{
return x_101;
}
else
{
lean_object* x_112; lean_object* x_113; 
x_112 = lean_ctor_get(x_101, 0);
lean_inc(x_112);
lean_dec(x_101);
x_113 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_113, 0, x_112);
return x_113;
}
}
}
else
{
lean_dec_ref(x_98);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_99;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_aesop_Aesop_RuleBuilder_default(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_Constructors(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_NormSimp(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_Tactic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_Apply(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Builder_Default(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Constructors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_NormSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0 = _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0();
lean_mark_persistent(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0);
lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1 = _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1();
lean_mark_persistent(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1);
lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2 = _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2();
lean_mark_persistent(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2);
lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3 = _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3();
lean_mark_persistent(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3);
lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4 = _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4();
lean_mark_persistent(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4);
lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5 = _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5();
lean_mark_persistent(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5);
lp_aesop_Aesop_RuleBuilder_default___closed__0 = _init_lp_aesop_Aesop_RuleBuilder_default___closed__0();
lean_mark_persistent(lp_aesop_Aesop_RuleBuilder_default___closed__0);
lp_aesop_Aesop_RuleBuilder_default___closed__1 = _init_lp_aesop_Aesop_RuleBuilder_default___closed__1();
lean_mark_persistent(lp_aesop_Aesop_RuleBuilder_default___closed__1);
lp_aesop_Aesop_RuleBuilder_default___closed__2 = _init_lp_aesop_Aesop_RuleBuilder_default___closed__2();
lean_mark_persistent(lp_aesop_Aesop_RuleBuilder_default___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
