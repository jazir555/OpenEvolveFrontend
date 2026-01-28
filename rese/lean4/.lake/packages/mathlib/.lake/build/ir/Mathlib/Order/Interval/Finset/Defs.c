// Lean compiler output
// Module: Mathlib.Order.Interval.Finset.Defs
// Imports: public import Init public import Mathlib.Data.Finset.Preimage public import Mathlib.Data.Finset.Prod public import Mathlib.Order.Hom.WithTopBot public import Mathlib.Order.Interval.Set.UnorderedInterval
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
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8;
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68;
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36;
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_subtype___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51;
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34;
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_coeWithTop___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57;
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42;
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18;
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72;
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0;
static lean_object* lp_mathlib_WithTop_insertTop___lam__0___closed__0;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13;
lean_object* l_String_toRawSubstring_x27(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52;
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0;
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21;
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83;
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60;
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14;
lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_toFinset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6;
lean_object* l_Lean_Name_mkStr1(lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg(lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10;
static lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9;
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31;
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1;
static lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37;
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69;
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 1;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
x_6 = lean_apply_2(x_2, x_3, x_4);
x_7 = lp_mathlib_Multiset_filter___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 1;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_3);
x_6 = lean_apply_2(x_2, x_3, x_4);
x_7 = lp_mathlib_Multiset_filter___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_1);
lean_inc(x_4);
x_5 = lean_apply_2(x_1, x_4, x_2);
x_6 = lean_unbox(x_5);
if (x_6 == 0)
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_apply_2(x_1, x_3, x_4);
x_8 = lean_unbox(x_7);
if (x_8 == 0)
{
uint8_t x_9; 
x_9 = 1;
return x_9;
}
else
{
uint8_t x_10; 
x_10 = lean_unbox(x_5);
return x_10;
}
}
else
{
uint8_t x_11; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_11 = 0;
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
lean_inc(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4___boxed), 4, 3);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
x_6 = lean_apply_2(x_2, x_3, x_4);
x_7 = lp_mathlib_Multiset_filter___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_2);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__1), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
lean_inc(x_2);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__3), 4, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
lean_inc(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__5), 4, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
x_6 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_6, 0, x_2);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_4);
lean_ctor_set(x_6, 3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_ofIcc_x27(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 1;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
x_6 = lean_apply_2(x_2, x_3, x_4);
x_7 = lp_mathlib_Multiset_filter___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
uint8_t x_6; 
x_6 = 1;
return x_6;
}
else
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_3);
x_6 = lean_apply_2(x_2, x_3, x_4);
x_7 = lp_mathlib_Multiset_filter___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_1);
lean_inc(x_4);
x_5 = lean_apply_2(x_1, x_2, x_4);
x_6 = lean_unbox(x_5);
if (x_6 == 0)
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_apply_2(x_1, x_4, x_3);
x_8 = lean_unbox(x_7);
if (x_8 == 0)
{
uint8_t x_9; 
x_9 = 1;
return x_9;
}
else
{
uint8_t x_10; 
x_10 = lean_unbox(x_5);
return x_10;
}
}
else
{
uint8_t x_11; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_11 = 0;
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
lean_inc(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4___boxed), 4, 3);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
x_6 = lean_apply_2(x_2, x_3, x_4);
x_7 = lp_mathlib_Multiset_filter___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_2);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__1), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
lean_inc(x_2);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__3), 4, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
lean_inc(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__5), 4, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
x_6 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_6, 0, x_2);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_4);
lean_ctor_set(x_6, 3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrder_ofIcc(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lp_mathlib_Multiset_filter___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg___lam__1), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderTop_ofIci_x27(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lp_mathlib_Multiset_filter___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg___lam__1), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderTop_ofIci(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lp_mathlib_Multiset_filter___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderBot_ofIic_x27(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_3);
x_5 = lean_apply_1(x_2, x_3);
x_6 = lp_mathlib_Multiset_filter___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg___lam__1), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LocallyFiniteOrderBot_ofIic(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0___boxed), 2, 0);
lean_inc_ref_n(x_4, 3);
x_5 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_4);
lean_ctor_set(x_5, 2, x_4);
lean_ctor_set(x_5, 3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsEmpty_toLocallyFiniteOrder(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0(lean_object* x_1) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0___boxed), 1, 0);
lean_inc_ref(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsEmpty_toLocallyFiniteOrderTop(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__1;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Icc___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Icc(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Ico___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Ico(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Ioc___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Ioc(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 3);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Ioo___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_Ioo(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Ici___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Ici(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Ioi___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Ioi(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Iic___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Iic(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Iio___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_Iio(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_Icc___redArg(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Finset_Ioc___redArg(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc(x_2);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_2);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Finset_Ico___boxed), 5, 4);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_3);
lean_closure_set(x_4, 3, x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Finset_Icc___boxed), 5, 4);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_3);
lean_closure_set(x_5, 3, x_2);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
lean_inc(x_4);
lean_inc(x_3);
x_8 = lean_apply_2(x_6, x_3, x_4);
x_9 = lean_apply_2(x_7, x_3, x_4);
x_10 = lp_mathlib_Finset_Icc___redArg(x_2, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_uIcc___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("FinsetInterval", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("term[[_,_]]", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1;
x_2 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("andthen", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("[[", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("term", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8;
x_3 = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9;
x_2 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6;
x_3 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(", ", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12;
x_2 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10;
x_3 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9;
x_2 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13;
x_3 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("]]", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15;
x_2 = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16;
x_2 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14;
x_3 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4;
x_4 = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17;
x_2 = lean_unsigned_to_nat(1024u);
x_3 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2;
x_4 = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18;
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Lean", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Parser", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Term", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("app", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2;
x_3 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1;
x_4 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Finset.uIcc", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Finset", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("uIcc", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("null", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2;
lean_inc(x_1);
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec_ref(x_2);
lean_dec(x_1);
x_6 = lean_box(1);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 5);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lean_unsigned_to_nat(1u);
x_12 = l_Lean_Syntax_getArg(x_1, x_11);
x_13 = lean_unsigned_to_nat(3u);
x_14 = l_Lean_Syntax_getArg(x_1, x_13);
lean_dec(x_1);
x_15 = 0;
x_16 = l_Lean_SourceInfo_fromRef(x_10, x_15);
lean_dec(x_10);
x_17 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4;
x_18 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6;
x_19 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9;
x_20 = l_Lean_addMacroScope(x_8, x_19, x_9);
x_21 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11;
lean_inc(x_16);
x_22 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_22, 0, x_16);
lean_ctor_set(x_22, 1, x_18);
lean_ctor_set(x_22, 2, x_20);
lean_ctor_set(x_22, 3, x_21);
x_23 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13;
lean_inc(x_16);
x_24 = l_Lean_Syntax_node2(x_16, x_23, x_12, x_14);
x_25 = l_Lean_Syntax_node2(x_16, x_17, x_22, x_24);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_3);
return x_26;
}
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ident", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4;
lean_inc(x_1);
x_5 = l_Lean_Syntax_isOfKind(x_1, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_1);
x_6 = lean_box(0);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_8 = lean_unsigned_to_nat(0u);
x_9 = l_Lean_Syntax_getArg(x_1, x_8);
x_10 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1;
lean_inc(x_9);
x_11 = l_Lean_Syntax_isOfKind(x_9, x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; 
lean_dec(x_9);
lean_dec(x_1);
x_12 = lean_box(0);
x_13 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_3);
return x_13;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_14 = lean_unsigned_to_nat(1u);
x_15 = l_Lean_Syntax_getArg(x_1, x_14);
lean_dec(x_1);
x_16 = lean_unsigned_to_nat(2u);
lean_inc(x_15);
x_17 = l_Lean_Syntax_matchesNull(x_15, x_16);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; 
lean_dec(x_15);
lean_dec(x_9);
x_18 = lean_box(0);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_3);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_20 = l_Lean_Syntax_getArg(x_15, x_8);
x_21 = l_Lean_Syntax_getArg(x_15, x_14);
lean_dec(x_15);
x_22 = l_Lean_replaceRef(x_9, x_2);
lean_dec(x_9);
x_23 = 0;
x_24 = l_Lean_SourceInfo_fromRef(x_22, x_23);
lean_dec(x_22);
x_25 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2;
x_26 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5;
lean_inc(x_24);
x_27 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_27, 0, x_24);
lean_ctor_set(x_27, 1, x_26);
x_28 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11;
lean_inc(x_24);
x_29 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_29, 0, x_24);
lean_ctor_set(x_29, 1, x_28);
x_30 = lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15;
lean_inc(x_24);
x_31 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_31, 0, x_24);
lean_ctor_set(x_31, 1, x_30);
x_32 = l_Lean_Syntax_node5(x_24, x_25, x_27, x_20, x_29, x_21, x_31);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_3);
return x_33;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Elab_unsupportedSyntaxExceptionId;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg() {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__1;
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_9;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Mathlib", 7, 7);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Meta", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("setBuilder", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1;
x_3 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0;
x_4 = l_Lean_Name_mkStr3(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Batteries", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("ExtendedBinder", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("extBinder", 9, 9);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5;
x_3 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4;
x_4 = l_Lean_Name_mkStr3(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("binderIdent", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("binderPred≤_", 14, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("binderPred≥_", 14, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("binderPred<_", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("binderPred>_", 12, 12);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Finset.filter", 13, 13);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("filter", 6, 6);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("paren", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2;
x_3 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1;
x_4 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hygienicLParen", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2;
x_3 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1;
x_4 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("(", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("hygieneInfo", 11, 11);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("", 0, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Elab", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39;
x_3 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_4 = l_Lean_Name_mkStr3(x_3, x_2, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Function", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49;
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58;
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("fun", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2;
x_3 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1;
x_4 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("basicFun", 8, 8);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2;
x_3 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1;
x_4 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0;
x_5 = l_Lean_Name_mkStr4(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64() {
_start:
{
lean_object* x_1; 
x_1 = l_Array_mkArray0(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("↦", 3, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(")", 1, 1);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Finset.Ioi", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Ioi", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Finset.Iio", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Iio", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Finset.Ici", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Ici", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Finset.Iic", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85;
x_2 = l_String_toRawSubstring_x27(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Iic", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87;
x_2 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; uint8_t x_11; 
x_10 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3;
lean_inc(x_1);
x_11 = l_Lean_Syntax_isOfKind(x_1, x_10);
if (x_11 == 0)
{
lean_object* x_12; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_12 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_13 = lean_unsigned_to_nat(1u);
x_14 = l_Lean_Syntax_getArg(x_1, x_13);
x_15 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7;
lean_inc(x_14);
x_16 = l_Lean_Syntax_isOfKind(x_14, x_15);
if (x_16 == 0)
{
lean_object* x_17; 
lean_dec(x_14);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_17 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_18 = lean_unsigned_to_nat(0u);
x_19 = l_Lean_Syntax_getArg(x_14, x_18);
x_20 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9;
lean_inc(x_19);
x_21 = l_Lean_Syntax_isOfKind(x_19, x_20);
if (x_21 == 0)
{
lean_object* x_22; 
lean_dec(x_19);
lean_dec(x_14);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_22 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_22;
}
else
{
lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_23 = l_Lean_Syntax_getArg(x_19, x_18);
lean_dec(x_19);
x_24 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1;
lean_inc(x_23);
x_25 = l_Lean_Syntax_isOfKind(x_23, x_24);
if (x_25 == 0)
{
lean_object* x_26; 
lean_dec(x_23);
lean_dec(x_14);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_26 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_26;
}
else
{
lean_object* x_27; uint8_t x_28; 
x_27 = l_Lean_Syntax_getArg(x_14, x_13);
lean_dec(x_14);
lean_inc(x_27);
x_28 = l_Lean_Syntax_matchesNull(x_27, x_13);
if (x_28 == 0)
{
lean_object* x_29; 
lean_dec(x_27);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_29 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_29;
}
else
{
lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_30 = l_Lean_Syntax_getArg(x_27, x_18);
lean_dec(x_27);
x_31 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11;
lean_inc(x_30);
x_32 = l_Lean_Syntax_isOfKind(x_30, x_31);
if (x_32 == 0)
{
lean_object* x_33; uint8_t x_34; 
x_33 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13;
lean_inc(x_30);
x_34 = l_Lean_Syntax_isOfKind(x_30, x_33);
if (x_34 == 0)
{
lean_object* x_35; uint8_t x_36; 
x_35 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15;
lean_inc(x_30);
x_36 = l_Lean_Syntax_isOfKind(x_30, x_35);
if (x_36 == 0)
{
lean_object* x_37; uint8_t x_38; 
x_37 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17;
lean_inc(x_30);
x_38 = l_Lean_Syntax_isOfKind(x_30, x_37);
if (x_38 == 0)
{
lean_object* x_39; 
lean_dec(x_30);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_39 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_39;
}
else
{
lean_object* x_40; 
lean_inc(x_2);
x_40 = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_40) == 0)
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; uint8_t x_101; 
x_41 = lean_ctor_get(x_40, 0);
lean_inc(x_41);
lean_dec_ref(x_40);
x_42 = l_Lean_Syntax_getArg(x_30, x_13);
lean_dec(x_30);
x_43 = lean_unsigned_to_nat(3u);
x_44 = l_Lean_Syntax_getArg(x_1, x_43);
lean_dec(x_1);
x_101 = lean_unbox(x_41);
lean_dec(x_41);
if (x_101 == 0)
{
lean_object* x_102; uint8_t x_103; 
lean_dec(x_44);
lean_dec(x_42);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_102 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
x_103 = !lean_is_exclusive(x_102);
if (x_103 == 0)
{
return x_102;
}
else
{
lean_object* x_104; lean_object* x_105; 
x_104 = lean_ctor_get(x_102, 0);
lean_inc(x_104);
lean_dec(x_102);
x_105 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_105, 0, x_104);
return x_105;
}
}
else
{
x_45 = x_3;
x_46 = x_4;
x_47 = x_5;
x_48 = x_6;
x_49 = x_7;
x_50 = x_8;
x_51 = lean_box(0);
goto block_100;
}
block_100:
{
lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; 
x_52 = lean_ctor_get(x_49, 5);
x_53 = lean_ctor_get(x_49, 10);
x_54 = lean_ctor_get(x_49, 11);
x_55 = l_Lean_SourceInfo_fromRef(x_52, x_36);
x_56 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4;
x_57 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19;
x_58 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21;
lean_inc(x_54);
lean_inc(x_53);
x_59 = l_Lean_addMacroScope(x_53, x_58, x_54);
x_60 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23;
lean_inc(x_55);
x_61 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_61, 0, x_55);
lean_ctor_set(x_61, 1, x_57);
lean_ctor_set(x_61, 2, x_59);
lean_ctor_set(x_61, 3, x_60);
x_62 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13;
x_63 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25;
x_64 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27;
x_65 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28;
lean_inc(x_55);
x_66 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_66, 0, x_55);
lean_ctor_set(x_66, 1, x_65);
x_67 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30;
x_68 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32;
x_69 = lean_box(0);
lean_inc(x_54);
lean_inc(x_53);
x_70 = l_Lean_addMacroScope(x_53, x_69, x_54);
x_71 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59;
lean_inc(x_55);
x_72 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_72, 0, x_55);
lean_ctor_set(x_72, 1, x_68);
lean_ctor_set(x_72, 2, x_70);
lean_ctor_set(x_72, 3, x_71);
lean_inc(x_55);
x_73 = l_Lean_Syntax_node1(x_55, x_67, x_72);
lean_inc(x_55);
x_74 = l_Lean_Syntax_node2(x_55, x_64, x_66, x_73);
x_75 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60;
x_76 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61;
lean_inc(x_55);
x_77 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_77, 0, x_55);
lean_ctor_set(x_77, 1, x_75);
x_78 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63;
lean_inc(x_55);
x_79 = l_Lean_Syntax_node1(x_55, x_62, x_23);
x_80 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64;
lean_inc(x_55);
x_81 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_81, 0, x_55);
lean_ctor_set(x_81, 1, x_62);
lean_ctor_set(x_81, 2, x_80);
x_82 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65;
lean_inc(x_55);
x_83 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_83, 0, x_55);
lean_ctor_set(x_83, 1, x_82);
lean_inc(x_55);
x_84 = l_Lean_Syntax_node4(x_55, x_78, x_79, x_81, x_83, x_44);
lean_inc(x_55);
x_85 = l_Lean_Syntax_node2(x_55, x_76, x_77, x_84);
x_86 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66;
lean_inc(x_55);
x_87 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_87, 0, x_55);
lean_ctor_set(x_87, 1, x_86);
lean_inc_ref(x_87);
lean_inc(x_74);
lean_inc(x_55);
x_88 = l_Lean_Syntax_node3(x_55, x_63, x_74, x_85, x_87);
x_89 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68;
x_90 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70;
lean_inc(x_54);
lean_inc(x_53);
x_91 = l_Lean_addMacroScope(x_53, x_90, x_54);
x_92 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72;
lean_inc(x_55);
x_93 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_93, 0, x_55);
lean_ctor_set(x_93, 1, x_89);
lean_ctor_set(x_93, 2, x_91);
lean_ctor_set(x_93, 3, x_92);
lean_inc(x_55);
x_94 = l_Lean_Syntax_node1(x_55, x_62, x_42);
lean_inc(x_55);
x_95 = l_Lean_Syntax_node2(x_55, x_56, x_93, x_94);
lean_inc(x_55);
x_96 = l_Lean_Syntax_node3(x_55, x_63, x_74, x_95, x_87);
lean_inc(x_55);
x_97 = l_Lean_Syntax_node2(x_55, x_62, x_88, x_96);
x_98 = l_Lean_Syntax_node2(x_55, x_56, x_61, x_97);
x_99 = l_Lean_Elab_Term_elabTerm(x_98, x_2, x_28, x_28, x_45, x_46, x_47, x_48, x_49, x_50);
return x_99;
}
}
else
{
uint8_t x_106; 
lean_dec(x_30);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_106 = !lean_is_exclusive(x_40);
if (x_106 == 0)
{
return x_40;
}
else
{
lean_object* x_107; lean_object* x_108; 
x_107 = lean_ctor_get(x_40, 0);
lean_inc(x_107);
lean_dec(x_40);
x_108 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_108, 0, x_107);
return x_108;
}
}
}
}
else
{
lean_object* x_109; 
lean_inc(x_2);
x_109 = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_109) == 0)
{
lean_object* x_110; lean_object* x_111; lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; lean_object* x_120; uint8_t x_170; 
x_110 = lean_ctor_get(x_109, 0);
lean_inc(x_110);
lean_dec_ref(x_109);
x_111 = l_Lean_Syntax_getArg(x_30, x_13);
lean_dec(x_30);
x_112 = lean_unsigned_to_nat(3u);
x_113 = l_Lean_Syntax_getArg(x_1, x_112);
lean_dec(x_1);
x_170 = lean_unbox(x_110);
lean_dec(x_110);
if (x_170 == 0)
{
lean_object* x_171; uint8_t x_172; 
lean_dec(x_113);
lean_dec(x_111);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_171 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
x_172 = !lean_is_exclusive(x_171);
if (x_172 == 0)
{
return x_171;
}
else
{
lean_object* x_173; lean_object* x_174; 
x_173 = lean_ctor_get(x_171, 0);
lean_inc(x_173);
lean_dec(x_171);
x_174 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_174, 0, x_173);
return x_174;
}
}
else
{
x_114 = x_3;
x_115 = x_4;
x_116 = x_5;
x_117 = x_6;
x_118 = x_7;
x_119 = x_8;
x_120 = lean_box(0);
goto block_169;
}
block_169:
{
lean_object* x_121; lean_object* x_122; lean_object* x_123; lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; lean_object* x_136; lean_object* x_137; lean_object* x_138; lean_object* x_139; lean_object* x_140; lean_object* x_141; lean_object* x_142; lean_object* x_143; lean_object* x_144; lean_object* x_145; lean_object* x_146; lean_object* x_147; lean_object* x_148; lean_object* x_149; lean_object* x_150; lean_object* x_151; lean_object* x_152; lean_object* x_153; lean_object* x_154; lean_object* x_155; lean_object* x_156; lean_object* x_157; lean_object* x_158; lean_object* x_159; lean_object* x_160; lean_object* x_161; lean_object* x_162; lean_object* x_163; lean_object* x_164; lean_object* x_165; lean_object* x_166; lean_object* x_167; lean_object* x_168; 
x_121 = lean_ctor_get(x_118, 5);
x_122 = lean_ctor_get(x_118, 10);
x_123 = lean_ctor_get(x_118, 11);
x_124 = l_Lean_SourceInfo_fromRef(x_121, x_34);
x_125 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4;
x_126 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19;
x_127 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21;
lean_inc(x_123);
lean_inc(x_122);
x_128 = l_Lean_addMacroScope(x_122, x_127, x_123);
x_129 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23;
lean_inc(x_124);
x_130 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_130, 0, x_124);
lean_ctor_set(x_130, 1, x_126);
lean_ctor_set(x_130, 2, x_128);
lean_ctor_set(x_130, 3, x_129);
x_131 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13;
x_132 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25;
x_133 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27;
x_134 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28;
lean_inc(x_124);
x_135 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_135, 0, x_124);
lean_ctor_set(x_135, 1, x_134);
x_136 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30;
x_137 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32;
x_138 = lean_box(0);
lean_inc(x_123);
lean_inc(x_122);
x_139 = l_Lean_addMacroScope(x_122, x_138, x_123);
x_140 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59;
lean_inc(x_124);
x_141 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_141, 0, x_124);
lean_ctor_set(x_141, 1, x_137);
lean_ctor_set(x_141, 2, x_139);
lean_ctor_set(x_141, 3, x_140);
lean_inc(x_124);
x_142 = l_Lean_Syntax_node1(x_124, x_136, x_141);
lean_inc(x_124);
x_143 = l_Lean_Syntax_node2(x_124, x_133, x_135, x_142);
x_144 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60;
x_145 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61;
lean_inc(x_124);
x_146 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_146, 0, x_124);
lean_ctor_set(x_146, 1, x_144);
x_147 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63;
lean_inc(x_124);
x_148 = l_Lean_Syntax_node1(x_124, x_131, x_23);
x_149 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64;
lean_inc(x_124);
x_150 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_150, 0, x_124);
lean_ctor_set(x_150, 1, x_131);
lean_ctor_set(x_150, 2, x_149);
x_151 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65;
lean_inc(x_124);
x_152 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_152, 0, x_124);
lean_ctor_set(x_152, 1, x_151);
lean_inc(x_124);
x_153 = l_Lean_Syntax_node4(x_124, x_147, x_148, x_150, x_152, x_113);
lean_inc(x_124);
x_154 = l_Lean_Syntax_node2(x_124, x_145, x_146, x_153);
x_155 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66;
lean_inc(x_124);
x_156 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_156, 0, x_124);
lean_ctor_set(x_156, 1, x_155);
lean_inc_ref(x_156);
lean_inc(x_143);
lean_inc(x_124);
x_157 = l_Lean_Syntax_node3(x_124, x_132, x_143, x_154, x_156);
x_158 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74;
x_159 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76;
lean_inc(x_123);
lean_inc(x_122);
x_160 = l_Lean_addMacroScope(x_122, x_159, x_123);
x_161 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78;
lean_inc(x_124);
x_162 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_162, 0, x_124);
lean_ctor_set(x_162, 1, x_158);
lean_ctor_set(x_162, 2, x_160);
lean_ctor_set(x_162, 3, x_161);
lean_inc(x_124);
x_163 = l_Lean_Syntax_node1(x_124, x_131, x_111);
lean_inc(x_124);
x_164 = l_Lean_Syntax_node2(x_124, x_125, x_162, x_163);
lean_inc(x_124);
x_165 = l_Lean_Syntax_node3(x_124, x_132, x_143, x_164, x_156);
lean_inc(x_124);
x_166 = l_Lean_Syntax_node2(x_124, x_131, x_157, x_165);
x_167 = l_Lean_Syntax_node2(x_124, x_125, x_130, x_166);
x_168 = l_Lean_Elab_Term_elabTerm(x_167, x_2, x_28, x_28, x_114, x_115, x_116, x_117, x_118, x_119);
return x_168;
}
}
else
{
uint8_t x_175; 
lean_dec(x_30);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_175 = !lean_is_exclusive(x_109);
if (x_175 == 0)
{
return x_109;
}
else
{
lean_object* x_176; lean_object* x_177; 
x_176 = lean_ctor_get(x_109, 0);
lean_inc(x_176);
lean_dec(x_109);
x_177 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_177, 0, x_176);
return x_177;
}
}
}
}
else
{
lean_object* x_178; 
lean_inc(x_2);
x_178 = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_178) == 0)
{
lean_object* x_179; lean_object* x_180; lean_object* x_181; lean_object* x_182; lean_object* x_183; lean_object* x_184; lean_object* x_185; lean_object* x_186; lean_object* x_187; lean_object* x_188; lean_object* x_189; uint8_t x_239; 
x_179 = lean_ctor_get(x_178, 0);
lean_inc(x_179);
lean_dec_ref(x_178);
x_180 = l_Lean_Syntax_getArg(x_30, x_13);
lean_dec(x_30);
x_181 = lean_unsigned_to_nat(3u);
x_182 = l_Lean_Syntax_getArg(x_1, x_181);
lean_dec(x_1);
x_239 = lean_unbox(x_179);
lean_dec(x_179);
if (x_239 == 0)
{
lean_object* x_240; uint8_t x_241; 
lean_dec(x_182);
lean_dec(x_180);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_240 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
x_241 = !lean_is_exclusive(x_240);
if (x_241 == 0)
{
return x_240;
}
else
{
lean_object* x_242; lean_object* x_243; 
x_242 = lean_ctor_get(x_240, 0);
lean_inc(x_242);
lean_dec(x_240);
x_243 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_243, 0, x_242);
return x_243;
}
}
else
{
x_183 = x_3;
x_184 = x_4;
x_185 = x_5;
x_186 = x_6;
x_187 = x_7;
x_188 = x_8;
x_189 = lean_box(0);
goto block_238;
}
block_238:
{
lean_object* x_190; lean_object* x_191; lean_object* x_192; lean_object* x_193; lean_object* x_194; lean_object* x_195; lean_object* x_196; lean_object* x_197; lean_object* x_198; lean_object* x_199; lean_object* x_200; lean_object* x_201; lean_object* x_202; lean_object* x_203; lean_object* x_204; lean_object* x_205; lean_object* x_206; lean_object* x_207; lean_object* x_208; lean_object* x_209; lean_object* x_210; lean_object* x_211; lean_object* x_212; lean_object* x_213; lean_object* x_214; lean_object* x_215; lean_object* x_216; lean_object* x_217; lean_object* x_218; lean_object* x_219; lean_object* x_220; lean_object* x_221; lean_object* x_222; lean_object* x_223; lean_object* x_224; lean_object* x_225; lean_object* x_226; lean_object* x_227; lean_object* x_228; lean_object* x_229; lean_object* x_230; lean_object* x_231; lean_object* x_232; lean_object* x_233; lean_object* x_234; lean_object* x_235; lean_object* x_236; lean_object* x_237; 
x_190 = lean_ctor_get(x_187, 5);
x_191 = lean_ctor_get(x_187, 10);
x_192 = lean_ctor_get(x_187, 11);
x_193 = l_Lean_SourceInfo_fromRef(x_190, x_32);
x_194 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4;
x_195 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19;
x_196 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21;
lean_inc(x_192);
lean_inc(x_191);
x_197 = l_Lean_addMacroScope(x_191, x_196, x_192);
x_198 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23;
lean_inc(x_193);
x_199 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_199, 0, x_193);
lean_ctor_set(x_199, 1, x_195);
lean_ctor_set(x_199, 2, x_197);
lean_ctor_set(x_199, 3, x_198);
x_200 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13;
x_201 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25;
x_202 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27;
x_203 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28;
lean_inc(x_193);
x_204 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_204, 0, x_193);
lean_ctor_set(x_204, 1, x_203);
x_205 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30;
x_206 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32;
x_207 = lean_box(0);
lean_inc(x_192);
lean_inc(x_191);
x_208 = l_Lean_addMacroScope(x_191, x_207, x_192);
x_209 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59;
lean_inc(x_193);
x_210 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_210, 0, x_193);
lean_ctor_set(x_210, 1, x_206);
lean_ctor_set(x_210, 2, x_208);
lean_ctor_set(x_210, 3, x_209);
lean_inc(x_193);
x_211 = l_Lean_Syntax_node1(x_193, x_205, x_210);
lean_inc(x_193);
x_212 = l_Lean_Syntax_node2(x_193, x_202, x_204, x_211);
x_213 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60;
x_214 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61;
lean_inc(x_193);
x_215 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_215, 0, x_193);
lean_ctor_set(x_215, 1, x_213);
x_216 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63;
lean_inc(x_193);
x_217 = l_Lean_Syntax_node1(x_193, x_200, x_23);
x_218 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64;
lean_inc(x_193);
x_219 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_219, 0, x_193);
lean_ctor_set(x_219, 1, x_200);
lean_ctor_set(x_219, 2, x_218);
x_220 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65;
lean_inc(x_193);
x_221 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_221, 0, x_193);
lean_ctor_set(x_221, 1, x_220);
lean_inc(x_193);
x_222 = l_Lean_Syntax_node4(x_193, x_216, x_217, x_219, x_221, x_182);
lean_inc(x_193);
x_223 = l_Lean_Syntax_node2(x_193, x_214, x_215, x_222);
x_224 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66;
lean_inc(x_193);
x_225 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_225, 0, x_193);
lean_ctor_set(x_225, 1, x_224);
lean_inc_ref(x_225);
lean_inc(x_212);
lean_inc(x_193);
x_226 = l_Lean_Syntax_node3(x_193, x_201, x_212, x_223, x_225);
x_227 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80;
x_228 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82;
lean_inc(x_192);
lean_inc(x_191);
x_229 = l_Lean_addMacroScope(x_191, x_228, x_192);
x_230 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84;
lean_inc(x_193);
x_231 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_231, 0, x_193);
lean_ctor_set(x_231, 1, x_227);
lean_ctor_set(x_231, 2, x_229);
lean_ctor_set(x_231, 3, x_230);
lean_inc(x_193);
x_232 = l_Lean_Syntax_node1(x_193, x_200, x_180);
lean_inc(x_193);
x_233 = l_Lean_Syntax_node2(x_193, x_194, x_231, x_232);
lean_inc(x_193);
x_234 = l_Lean_Syntax_node3(x_193, x_201, x_212, x_233, x_225);
lean_inc(x_193);
x_235 = l_Lean_Syntax_node2(x_193, x_200, x_226, x_234);
x_236 = l_Lean_Syntax_node2(x_193, x_194, x_199, x_235);
x_237 = l_Lean_Elab_Term_elabTerm(x_236, x_2, x_28, x_28, x_183, x_184, x_185, x_186, x_187, x_188);
return x_237;
}
}
else
{
uint8_t x_244; 
lean_dec(x_30);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_244 = !lean_is_exclusive(x_178);
if (x_244 == 0)
{
return x_178;
}
else
{
lean_object* x_245; lean_object* x_246; 
x_245 = lean_ctor_get(x_178, 0);
lean_inc(x_245);
lean_dec(x_178);
x_246 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_246, 0, x_245);
return x_246;
}
}
}
}
else
{
lean_object* x_247; 
lean_inc(x_2);
x_247 = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(x_2, x_3, x_4, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_247) == 0)
{
lean_object* x_248; lean_object* x_249; lean_object* x_250; lean_object* x_251; lean_object* x_252; lean_object* x_253; lean_object* x_254; lean_object* x_255; lean_object* x_256; lean_object* x_257; lean_object* x_258; uint8_t x_309; 
x_248 = lean_ctor_get(x_247, 0);
lean_inc(x_248);
lean_dec_ref(x_247);
x_249 = l_Lean_Syntax_getArg(x_30, x_13);
lean_dec(x_30);
x_250 = lean_unsigned_to_nat(3u);
x_251 = l_Lean_Syntax_getArg(x_1, x_250);
lean_dec(x_1);
x_309 = lean_unbox(x_248);
lean_dec(x_248);
if (x_309 == 0)
{
lean_object* x_310; uint8_t x_311; 
lean_dec(x_251);
lean_dec(x_249);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
x_310 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
x_311 = !lean_is_exclusive(x_310);
if (x_311 == 0)
{
return x_310;
}
else
{
lean_object* x_312; lean_object* x_313; 
x_312 = lean_ctor_get(x_310, 0);
lean_inc(x_312);
lean_dec(x_310);
x_313 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_313, 0, x_312);
return x_313;
}
}
else
{
x_252 = x_3;
x_253 = x_4;
x_254 = x_5;
x_255 = x_6;
x_256 = x_7;
x_257 = x_8;
x_258 = lean_box(0);
goto block_308;
}
block_308:
{
lean_object* x_259; lean_object* x_260; lean_object* x_261; uint8_t x_262; lean_object* x_263; lean_object* x_264; lean_object* x_265; lean_object* x_266; lean_object* x_267; lean_object* x_268; lean_object* x_269; lean_object* x_270; lean_object* x_271; lean_object* x_272; lean_object* x_273; lean_object* x_274; lean_object* x_275; lean_object* x_276; lean_object* x_277; lean_object* x_278; lean_object* x_279; lean_object* x_280; lean_object* x_281; lean_object* x_282; lean_object* x_283; lean_object* x_284; lean_object* x_285; lean_object* x_286; lean_object* x_287; lean_object* x_288; lean_object* x_289; lean_object* x_290; lean_object* x_291; lean_object* x_292; lean_object* x_293; lean_object* x_294; lean_object* x_295; lean_object* x_296; lean_object* x_297; lean_object* x_298; lean_object* x_299; lean_object* x_300; lean_object* x_301; lean_object* x_302; lean_object* x_303; lean_object* x_304; lean_object* x_305; lean_object* x_306; lean_object* x_307; 
x_259 = lean_ctor_get(x_256, 5);
x_260 = lean_ctor_get(x_256, 10);
x_261 = lean_ctor_get(x_256, 11);
x_262 = 0;
x_263 = l_Lean_SourceInfo_fromRef(x_259, x_262);
x_264 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4;
x_265 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19;
x_266 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21;
lean_inc(x_261);
lean_inc(x_260);
x_267 = l_Lean_addMacroScope(x_260, x_266, x_261);
x_268 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23;
lean_inc(x_263);
x_269 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_269, 0, x_263);
lean_ctor_set(x_269, 1, x_265);
lean_ctor_set(x_269, 2, x_267);
lean_ctor_set(x_269, 3, x_268);
x_270 = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13;
x_271 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25;
x_272 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27;
x_273 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28;
lean_inc(x_263);
x_274 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_274, 0, x_263);
lean_ctor_set(x_274, 1, x_273);
x_275 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30;
x_276 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32;
x_277 = lean_box(0);
lean_inc(x_261);
lean_inc(x_260);
x_278 = l_Lean_addMacroScope(x_260, x_277, x_261);
x_279 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59;
lean_inc(x_263);
x_280 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_280, 0, x_263);
lean_ctor_set(x_280, 1, x_276);
lean_ctor_set(x_280, 2, x_278);
lean_ctor_set(x_280, 3, x_279);
lean_inc(x_263);
x_281 = l_Lean_Syntax_node1(x_263, x_275, x_280);
lean_inc(x_263);
x_282 = l_Lean_Syntax_node2(x_263, x_272, x_274, x_281);
x_283 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60;
x_284 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61;
lean_inc(x_263);
x_285 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_285, 0, x_263);
lean_ctor_set(x_285, 1, x_283);
x_286 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63;
lean_inc(x_263);
x_287 = l_Lean_Syntax_node1(x_263, x_270, x_23);
x_288 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64;
lean_inc(x_263);
x_289 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_289, 0, x_263);
lean_ctor_set(x_289, 1, x_270);
lean_ctor_set(x_289, 2, x_288);
x_290 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65;
lean_inc(x_263);
x_291 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_291, 0, x_263);
lean_ctor_set(x_291, 1, x_290);
lean_inc(x_263);
x_292 = l_Lean_Syntax_node4(x_263, x_286, x_287, x_289, x_291, x_251);
lean_inc(x_263);
x_293 = l_Lean_Syntax_node2(x_263, x_284, x_285, x_292);
x_294 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66;
lean_inc(x_263);
x_295 = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(x_295, 0, x_263);
lean_ctor_set(x_295, 1, x_294);
lean_inc_ref(x_295);
lean_inc(x_282);
lean_inc(x_263);
x_296 = l_Lean_Syntax_node3(x_263, x_271, x_282, x_293, x_295);
x_297 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86;
x_298 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88;
lean_inc(x_261);
lean_inc(x_260);
x_299 = l_Lean_addMacroScope(x_260, x_298, x_261);
x_300 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90;
lean_inc(x_263);
x_301 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_301, 0, x_263);
lean_ctor_set(x_301, 1, x_297);
lean_ctor_set(x_301, 2, x_299);
lean_ctor_set(x_301, 3, x_300);
lean_inc(x_263);
x_302 = l_Lean_Syntax_node1(x_263, x_270, x_249);
lean_inc(x_263);
x_303 = l_Lean_Syntax_node2(x_263, x_264, x_301, x_302);
lean_inc(x_263);
x_304 = l_Lean_Syntax_node3(x_263, x_271, x_282, x_303, x_295);
lean_inc(x_263);
x_305 = l_Lean_Syntax_node2(x_263, x_270, x_296, x_304);
x_306 = l_Lean_Syntax_node2(x_263, x_264, x_269, x_305);
x_307 = l_Lean_Elab_Term_elabTerm(x_306, x_2, x_28, x_28, x_252, x_253, x_254, x_255, x_256, x_257);
return x_307;
}
}
else
{
uint8_t x_314; 
lean_dec(x_30);
lean_dec(x_23);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec(x_1);
x_314 = !lean_is_exclusive(x_247);
if (x_314 == 0)
{
return x_247;
}
else
{
lean_object* x_315; lean_object* x_316; 
x_315 = lean_ctor_get(x_247, 0);
lean_inc(x_315);
lean_dec(x_247);
x_316 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_316, 0, x_315);
return x_316;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Icc___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Fintype_subtype___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIcc___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIcc(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ico___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Fintype_subtype___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIco___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIco(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ioc___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Fintype_subtype___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIoc___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIoc(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ioo___redArg(x_1, x_2, x_3);
x_5 = lp_mathlib_Fintype_subtype___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIoo___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_instFintypeIoo(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Finset_Ici___redArg(x_1, x_2);
x_4 = lp_mathlib_Fintype_subtype___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIci___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIci(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Finset_Ioi___redArg(x_1, x_2);
x_4 = lp_mathlib_Fintype_subtype___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIoi___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIoi(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Finset_Iic___redArg(x_1, x_2);
x_4 = lp_mathlib_Fintype_subtype___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIic___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIic(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Finset_Iio___redArg(x_1, x_2);
x_4 = lp_mathlib_Fintype_subtype___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIio___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Set_instFintypeIio(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_uIcc___redArg(x_1, x_2, x_3, x_4);
x_6 = lp_mathlib_Fintype_subtype___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Set_fintypeUIcc___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_1);
lean_inc(x_4);
x_5 = lean_apply_2(x_1, x_2, x_4);
x_6 = lean_unbox(x_5);
if (x_6 == 0)
{
uint8_t x_7; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_1);
x_7 = lean_unbox(x_5);
return x_7;
}
else
{
lean_object* x_8; uint8_t x_9; 
x_8 = lean_apply_2(x_1, x_4, x_3);
x_9 = lean_unbox(x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_4);
x_6 = lp_mathlib_Subtype_fintype___redArg(x_5, x_2);
x_7 = lp_mathlib_Set_toFinset___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_5);
x_6 = lean_apply_2(x_1, x_2, x_5);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
uint8_t x_8; 
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
x_8 = lean_unbox(x_6);
return x_8;
}
else
{
lean_object* x_9; uint8_t x_10; 
x_9 = lean_apply_2(x_3, x_5, x_4);
x_10 = lean_unbox(x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2(x_1, x_2, x_3, x_4, x_5);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2___boxed), 5, 4);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_4);
lean_closure_set(x_6, 2, x_2);
lean_closure_set(x_6, 3, x_5);
x_7 = lp_mathlib_Subtype_fintype___redArg(x_6, x_3);
x_8 = lp_mathlib_Set_toFinset___redArg(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_3);
lean_inc_ref(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_3);
lean_inc(x_3);
lean_inc_ref(x_4);
lean_inc_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_3);
lean_inc(x_3);
lean_inc_ref(x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(x_8, 0, x_4);
lean_closure_set(x_8, 1, x_5);
lean_closure_set(x_8, 2, x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_9, 0, x_4);
lean_closure_set(x_9, 1, x_3);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_6);
lean_ctor_set(x_10, 1, x_7);
lean_ctor_set(x_10, 2, x_8);
lean_ctor_set(x_10, 3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_1);
lean_inc_ref(x_3);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_4, 0, x_3);
lean_closure_set(x_4, 1, x_1);
lean_inc(x_1);
lean_inc_ref(x_2);
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_1);
lean_inc(x_1);
lean_inc_ref(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_3);
lean_closure_set(x_6, 2, x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_1);
x_8 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_8, 0, x_4);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 2, x_6);
lean_ctor_set(x_8, 3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Fintype_toLocallyFiniteOrder(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
static lean_object* _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lp_mathlib_Finset_Icc___redArg(x_1, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lp_mathlib_Finset_Ioc___redArg(x_1, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lp_mathlib_Finset_Ico___redArg(x_1, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lp_mathlib_Finset_Ioo___redArg(x_1, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1), 3, 1);
lean_closure_set(x_3, 0, x_1);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__2), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__3), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_6, 0, x_2);
lean_ctor_set(x_6, 1, x_3);
lean_ctor_set(x_6, 2, x_4);
lean_ctor_set(x_6, 3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrder(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lp_mathlib_Finset_Ioi___redArg(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lp_mathlib_Finset_Ici___redArg(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrderBot(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lp_mathlib_Finset_Iio___redArg(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lp_mathlib_Finset_Iic___redArg(x_1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_OrderDual_instLocallyFiniteOrderTop(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lean_ctor_get(x_4, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_4, 1);
lean_inc(x_8);
lean_dec_ref(x_4);
x_9 = lp_mathlib_Finset_Icc___redArg(x_1, x_5, x_7);
x_10 = lp_mathlib_Finset_Icc___redArg(x_2, x_6, x_8);
x_11 = lp_mathlib_Multiset_product___redArg(x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Prod_instLocallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Prod_instLocallyFiniteOrder___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Prod_instLocallyFiniteOrder(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_Finset_Ici___redArg(x_1, x_4);
x_7 = lp_mathlib_Finset_Ici___redArg(x_2, x_5);
x_8 = lp_mathlib_Multiset_product___redArg(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Prod_instLocallyFiniteOrderTop(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lp_mathlib_Finset_Iic___redArg(x_1, x_4);
x_7 = lp_mathlib_Finset_Iic___redArg(x_2, x_5);
x_8 = lp_mathlib_Multiset_product___redArg(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Prod_instLocallyFiniteOrderBot(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
static lean_object* _init_lp_mathlib_WithTop_insertTop___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_coeWithTop___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_4 = lp_mathlib_Finset_map___redArg(x_3, x_2);
x_5 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_box(0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_WithTop_insertTop___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec_ref(x_3);
lean_dec_ref(x_2);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_box(0);
x_7 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_7, 0, x_1);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
else
{
lean_object* x_8; 
lean_dec_ref(x_5);
lean_dec(x_1);
x_8 = lean_box(0);
return x_8;
}
}
else
{
lean_dec(x_1);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec_ref(x_3);
x_9 = lean_ctor_get(x_4, 0);
lean_inc(x_9);
lean_dec_ref(x_4);
x_10 = lp_mathlib_Finset_Ici___redArg(x_2, x_9);
x_11 = lp_mathlib_WithTop_insertTop(lean_box(0));
x_12 = lean_apply_1(x_11, x_10);
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_dec_ref(x_2);
x_13 = lean_ctor_get(x_4, 0);
lean_inc(x_13);
lean_dec_ref(x_4);
x_14 = lean_ctor_get(x_5, 0);
lean_inc(x_14);
lean_dec_ref(x_5);
x_15 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_16 = lp_mathlib_Finset_Icc___redArg(x_3, x_13, x_14);
x_17 = lp_mathlib_Finset_map___redArg(x_15, x_16);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_5 = lean_box(0);
return x_5;
}
else
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec_ref(x_2);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_8 = lp_mathlib_Finset_Ici___redArg(x_1, x_6);
x_9 = lp_mathlib_Finset_map___redArg(x_7, x_8);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_1);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
lean_dec_ref(x_3);
x_11 = lean_ctor_get(x_4, 0);
lean_inc(x_11);
lean_dec_ref(x_4);
x_12 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_13 = lp_mathlib_Finset_Ico___redArg(x_2, x_10, x_11);
x_14 = lp_mathlib_Finset_map___redArg(x_12, x_13);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_5 = lean_box(0);
return x_5;
}
else
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec_ref(x_2);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lp_mathlib_Finset_Ioi___redArg(x_1, x_6);
x_8 = lp_mathlib_WithTop_insertTop(lean_box(0));
x_9 = lean_apply_1(x_8, x_7);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_1);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
lean_dec_ref(x_3);
x_11 = lean_ctor_get(x_4, 0);
lean_inc(x_11);
lean_dec_ref(x_4);
x_12 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_13 = lp_mathlib_Finset_Ioc___redArg(x_2, x_10, x_11);
x_14 = lp_mathlib_Finset_map___redArg(x_12, x_13);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_5 = lean_box(0);
return x_5;
}
else
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_dec_ref(x_2);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_8 = lp_mathlib_Finset_Ioi___redArg(x_1, x_6);
x_9 = lp_mathlib_Finset_map___redArg(x_7, x_8);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_1);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
lean_dec_ref(x_3);
x_11 = lean_ctor_get(x_4, 0);
lean_inc(x_11);
lean_dec_ref(x_4);
x_12 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_13 = lp_mathlib_Finset_Ioo___redArg(x_2, x_10, x_11);
x_14 = lp_mathlib_Finset_map___redArg(x_12, x_13);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lean_box(0);
lean_inc_ref(x_2);
x_4 = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(x_2, x_1);
lean_inc_ref(x_2);
lean_inc_ref(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__0), 5, 3);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
lean_closure_set(x_5, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(x_7, 0, x_4);
lean_closure_set(x_7, 1, x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_WithTop_locallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(x_8, 0, x_4);
lean_closure_set(x_8, 1, x_2);
x_9 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_9, 0, x_5);
lean_ctor_set(x_9, 1, x_6);
lean_ctor_set(x_9, 2, x_7);
lean_ctor_set(x_9, 3, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_WithTop_locallyFiniteOrder___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_locallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_WithTop_locallyFiniteOrder(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_box(0);
x_3 = lp_mathlib_WithTop_insertTop___lam__0___closed__0;
x_4 = lp_mathlib_Finset_map___redArg(x_3, x_1);
x_5 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_WithBot_insertBot___lam__0), 1, 0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(x_2);
x_4 = lp_mathlib_WithTop_locallyFiniteOrder___redArg(x_1, x_3);
x_5 = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_WithBot_instLocallyFiniteOrder___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_WithBot_instLocallyFiniteOrder(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_Equiv_symm___redArg(x_1);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_5);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_8 = lean_apply_1(x_7, x_3);
x_9 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_10 = lean_apply_1(x_9, x_4);
x_11 = lp_mathlib_Finset_Icc___redArg(x_2, x_8, x_10);
x_12 = lp_mathlib_Finset_map___redArg(x_6, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_Equiv_symm___redArg(x_1);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_5);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_8 = lean_apply_1(x_7, x_3);
x_9 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_10 = lean_apply_1(x_9, x_4);
x_11 = lp_mathlib_Finset_Ico___redArg(x_2, x_8, x_10);
x_12 = lp_mathlib_Finset_map___redArg(x_6, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_Equiv_symm___redArg(x_1);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_5);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_8 = lean_apply_1(x_7, x_3);
x_9 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_10 = lean_apply_1(x_9, x_4);
x_11 = lp_mathlib_Finset_Ioc___redArg(x_2, x_8, x_10);
x_12 = lp_mathlib_Finset_map___redArg(x_6, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_Equiv_symm___redArg(x_1);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_5);
lean_inc_ref(x_1);
x_7 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_8 = lean_apply_1(x_7, x_3);
x_9 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_10 = lean_apply_1(x_9, x_4);
x_11 = lp_mathlib_Finset_Ioo___redArg(x_2, x_8, x_10);
x_12 = lp_mathlib_Finset_map___redArg(x_6, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_5);
lean_inc_ref(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(x_7, 0, x_6);
lean_closure_set(x_7, 1, x_5);
lean_inc_ref(x_5);
lean_inc_ref(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_8, 0, x_6);
lean_closure_set(x_8, 1, x_5);
lean_inc_ref(x_5);
lean_inc_ref(x_6);
x_9 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(x_9, 0, x_6);
lean_closure_set(x_9, 1, x_5);
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(x_10, 0, x_6);
lean_closure_set(x_10, 1, x_5);
x_11 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_11, 0, x_7);
lean_ctor_set(x_11, 1, x_8);
lean_ctor_set(x_11, 2, x_9);
lean_ctor_set(x_11, 3, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_1);
x_7 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_5);
lean_ctor_set(x_7, 3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_OrderIso_locallyFiniteOrder(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Equiv_symm___redArg(x_1);
x_5 = lp_mathlib_Equiv_toEmbedding___redArg(x_4);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_7 = lean_apply_1(x_6, x_3);
x_8 = lp_mathlib_Finset_Ioi___redArg(x_2, x_7);
x_9 = lp_mathlib_Finset_map___redArg(x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Equiv_symm___redArg(x_1);
x_5 = lp_mathlib_Equiv_toEmbedding___redArg(x_4);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_7 = lean_apply_1(x_6, x_3);
x_8 = lp_mathlib_Finset_Ici___redArg(x_2, x_7);
x_9 = lp_mathlib_Finset_map___redArg(x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_5);
lean_inc_ref(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_6);
lean_closure_set(x_7, 1, x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(x_8, 0, x_6);
lean_closure_set(x_8, 1, x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_OrderIso_locallyFiniteOrderTop(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Equiv_symm___redArg(x_1);
x_5 = lp_mathlib_Equiv_toEmbedding___redArg(x_4);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_7 = lean_apply_1(x_6, x_3);
x_8 = lp_mathlib_Finset_Iio___redArg(x_2, x_7);
x_9 = lp_mathlib_Finset_map___redArg(x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_1);
x_4 = lp_mathlib_Equiv_symm___redArg(x_1);
x_5 = lp_mathlib_Equiv_toEmbedding___redArg(x_4);
x_6 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_7 = lean_apply_1(x_6, x_3);
x_8 = lp_mathlib_Finset_Iic___redArg(x_2, x_7);
x_9 = lp_mathlib_Finset_map___redArg(x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_5);
lean_inc_ref(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_6);
lean_closure_set(x_7, 1, x_5);
x_8 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__1), 3, 2);
lean_closure_set(x_8, 0, x_6);
lean_closure_set(x_8, 1, x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__1), 3, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_OrderIso_locallyFiniteOrderBot(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_Icc___redArg(x_1, x_3, x_4);
x_6 = lp_mathlib_Finset_subtype___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_Ico___redArg(x_1, x_3, x_4);
x_6 = lp_mathlib_Finset_subtype___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_Ioc___redArg(x_1, x_3, x_4);
x_6 = lp_mathlib_Finset_subtype___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_Ioo___redArg(x_1, x_3, x_4);
x_6 = lp_mathlib_Finset_subtype___redArg(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_1);
x_7 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_5);
lean_ctor_set(x_7, 3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrder(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ioi___redArg(x_1, x_3);
x_5 = lp_mathlib_Finset_subtype___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Ici___redArg(x_1, x_3);
x_5 = lp_mathlib_Finset_subtype___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrderTop(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Iio___redArg(x_1, x_3);
x_5 = lp_mathlib_Finset_subtype___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Finset_Iic___redArg(x_1, x_3);
x_5 = lp_mathlib_Finset_subtype___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__1), 3, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrderBot(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(x_1, x_2);
x_6 = lp_mathlib_Finset_Icc___redArg(x_5, x_4, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(x_1, x_2);
x_6 = lp_mathlib_Finset_Ioc___redArg(x_5, x_4, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
lean_inc(x_1);
lean_inc_ref(x_3);
lean_inc_ref(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2), 4, 3);
lean_closure_set(x_5, 0, x_4);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_3);
lean_closure_set(x_6, 2, x_1);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_Ioo___redArg(x_1, x_4, x_2);
x_6 = lp_mathlib_Finset_subtype___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Finset_Ico___redArg(x_1, x_4, x_2);
x_6 = lp_mathlib_Finset_subtype___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
lean_inc_ref(x_4);
lean_inc(x_1);
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2), 4, 3);
lean_closure_set(x_6, 0, x_3);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_2);
x_5 = lean_apply_1(x_1, x_2);
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(x_5, x_3);
x_7 = lp_mathlib_Finset_Icc___redArg(x_6, x_2, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_2);
x_5 = lean_apply_1(x_1, x_2);
x_6 = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(x_5, x_3);
x_7 = lp_mathlib_Finset_Ico___redArg(x_6, x_2, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_3);
lean_inc(x_1);
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0), 4, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_2);
x_5 = lean_apply_1(x_1, x_2);
x_6 = lp_mathlib_Finset_Ioo___redArg(x_3, x_2, x_4);
x_7 = lp_mathlib_Finset_subtype___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_2);
x_5 = lean_apply_1(x_1, x_2);
x_6 = lp_mathlib_Finset_Ioc___redArg(x_3, x_2, x_4);
x_7 = lp_mathlib_Finset_subtype___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_3);
lean_inc(x_1);
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__0), 4, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_1);
lean_closure_set(x_4, 2, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(x_5, 0, x_2);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 0);
lean_inc(x_8);
lean_dec_ref(x_2);
lean_inc(x_3);
x_9 = lean_apply_1(x_7, x_3);
lean_inc(x_6);
lean_inc(x_3);
x_10 = lean_apply_2(x_6, x_3, x_4);
x_11 = lean_apply_2(x_6, x_3, x_5);
x_12 = lean_apply_2(x_8, x_10, x_11);
x_13 = lp_mathlib_Finset_map___redArg(x_9, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 1);
lean_inc(x_8);
lean_dec_ref(x_2);
lean_inc(x_3);
x_9 = lean_apply_1(x_7, x_3);
lean_inc(x_6);
lean_inc(x_3);
x_10 = lean_apply_2(x_6, x_3, x_4);
x_11 = lean_apply_2(x_6, x_3, x_5);
x_12 = lean_apply_2(x_8, x_10, x_11);
x_13 = lp_mathlib_Finset_map___redArg(x_9, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 2);
lean_inc(x_8);
lean_dec_ref(x_2);
lean_inc(x_3);
x_9 = lean_apply_1(x_7, x_3);
lean_inc(x_6);
lean_inc(x_3);
x_10 = lean_apply_2(x_6, x_3, x_4);
x_11 = lean_apply_2(x_6, x_3, x_5);
x_12 = lean_apply_2(x_8, x_10, x_11);
x_13 = lp_mathlib_Finset_map___redArg(x_9, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 3);
lean_inc(x_8);
lean_dec_ref(x_2);
lean_inc(x_3);
x_9 = lean_apply_1(x_7, x_3);
lean_inc(x_6);
lean_inc(x_3);
x_10 = lean_apply_2(x_6, x_3, x_4);
x_11 = lean_apply_2(x_6, x_3, x_5);
x_12 = lean_apply_2(x_8, x_10, x_11);
x_13 = lp_mathlib_Finset_map___redArg(x_9, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_inc(x_8);
lean_inc_ref(x_9);
lean_inc_ref(x_6);
x_10 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0), 5, 3);
lean_closure_set(x_10, 0, x_6);
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_8);
lean_inc(x_8);
lean_inc_ref(x_9);
lean_inc_ref(x_6);
x_11 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1), 5, 3);
lean_closure_set(x_11, 0, x_6);
lean_closure_set(x_11, 1, x_9);
lean_closure_set(x_11, 2, x_8);
lean_inc(x_8);
lean_inc_ref(x_9);
lean_inc_ref(x_6);
x_12 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2), 5, 3);
lean_closure_set(x_12, 0, x_6);
lean_closure_set(x_12, 1, x_9);
lean_closure_set(x_12, 2, x_8);
x_13 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3), 5, 3);
lean_closure_set(x_13, 0, x_6);
lean_closure_set(x_13, 1, x_9);
lean_closure_set(x_13, 2, x_8);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_10);
lean_ctor_set(x_14, 1, x_11);
lean_ctor_set(x_14, 2, x_12);
lean_ctor_set(x_14, 3, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_2);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_2);
lean_inc(x_2);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1), 5, 3);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_3);
lean_closure_set(x_5, 2, x_2);
lean_inc(x_2);
lean_inc_ref(x_3);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2), 5, 3);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_3);
lean_closure_set(x_6, 2, x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3), 5, 3);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_2);
x_8 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_8, 0, x_4);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 2, x_6);
lean_ctor_set(x_8, 3, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_WithTopBot(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_WithTopBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0 = _init_lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0();
lean_mark_persistent(lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0);
lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__1 = _init_lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__1();
lean_mark_persistent(lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__1);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18 = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18);
lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d = _init_lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d();
lean_mark_persistent(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0);
lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1 = _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1();
lean_mark_persistent(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1);
lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0 = _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0);
lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__1 = _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__1);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89);
lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90 = _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90);
lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0 = _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0___closed__0);
lp_mathlib_WithTop_insertTop___lam__0___closed__0 = _init_lp_mathlib_WithTop_insertTop___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_WithTop_insertTop___lam__0___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
