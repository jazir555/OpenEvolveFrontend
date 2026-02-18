import ie, { useState as x, useEffect as C } from "react";
let O = {
  config: {},
  lastVizUrl: null,
  isInitializing: !1,
  error: null,
  features: {
    pygraphistryEnabled: !1,
    causalDiscoveryEnabled: !1,
    optimizationEnabled: !1,
    uqEnabled: !1,
    globalChemEnabled: !1,
    curieEnabled: !1,
    temporalGraphEnabled: !1,
    onekeEnabled: !1,
    leanAideEnabled: !1,
    sopEnabled: !1,
    adversarialEnabled: !1,
    pamiEnabled: !1,
    aceEnabled: !1,
    romaEnabled: !1,
    datapizzaEnabled: !1,
    crewaiEnabled: !1,
    claudiomiroEnabled: !1,
    steerEnabled: !1,
    researchQuestEnabled: !1,
    kgEnabled: !1,
    sgdEnabled: !1,
    globalAnalyticsEnabled: !1,
    mapElitesEnabled: !1,
    verificationEnabled: !1,
    problemAnalysisEnabled: !1,
    dependencyEnabled: !1,
    artifactGraphEnabled: !1,
    sceEnabled: !1,
    staticAnalysisEnabled: !1,
    lltlEnabled: !1,
    collaborationEnabled: !1,
    workflowMonitorEnabled: !1,
    lineageEnabled: !1,
    gauntletEnabled: !1,
    patternMiningEnabled: !1,
    adaptationEnabled: !1,
    ditoEnabled: !1,
    crewaiEnabled: !1,
    ragEnabled: !1,
    deepkeEnabled: !1,
    lean4Enabled: !1,
    makerEnabled: !1,
    mdapEnabled: !1,
    mctsEnabled: !1,
    hybridMCTSEnabled: !1,
    e2ePlannerEnabled: !1,
    evaluatorTeamEnabled: !1,
    redTeamEnabled: !1,
    blueTeamEnabled: !1,
    qaSuiteEnabled: !1,
    reseEnabled: !1,
    materialKGEnabled: !1,
    gnomeEnabled: !1,
    physicsNemoEnabled: !1,
    autogptEnabled: !1,
    autogenEnabled: !1,
    metagptEnabled: !1,
    llm4iasEnabled: !1,
    claraverseEnabled: !1,
    aiScientistEnabled: !1,
    uncertainpyEnabled: !1,
    riskAnalyzerEnabled: !1,
    karateclubEnabled: !1,
    neuralKGEnabled: !1,
    pylabrobotEnabled: !1,
    pinnsEnabled: !1
  }
};
class ps {
  async fetchVisualizationUrl(m) {
    if (!O.features.pygraphistryEnabled)
      throw new Error("PyGraphistry visualization is currently disabled.");
    try {
      const i = await fetch("/api/openevolve/visualize/pygraphistry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...m,
          config: O.config
        })
      });
      if (!i.ok) {
        const n = await i.json();
        throw new Error(n.detail || "Failed to generate PyGraphistry visualization");
      }
      return (await i.json()).url;
    } catch (i) {
      throw i;
    }
  }
}
function hs() {
  const t = new ps();
  return {
    initialize: async (m) => {
      O.isInitializing = !0, O.config = m, O.isInitializing = !1;
    },
    generateVisualization: async (m) => {
      try {
        const i = await t.fetchVisualizationUrl(m);
        return O.lastVizUrl = i, i;
      } catch (i) {
        return O.error = i instanceof Error ? i.message : String(i), null;
      }
    },
    updateFeatures: (m) => {
      O.features = { ...O.features, ...m };
    },
    updateConfig: (m) => {
      O.config = { ...O.config, ...m };
    },
    getState: () => ({ ...O })
  };
}
const v = hs();
var ne = { exports: {} }, q = {};
/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var Pe;
function bs() {
  if (Pe) return q;
  Pe = 1;
  var t = ie, m = Symbol.for("react.element"), i = Symbol.for("react.fragment"), h = Object.prototype.hasOwnProperty, n = t.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, d = { key: !0, ref: !0, __self: !0, __source: !0 };
  function o(b, p, s) {
    var r, c = {}, a = null, u = null;
    s !== void 0 && (a = "" + s), p.key !== void 0 && (a = "" + p.key), p.ref !== void 0 && (u = p.ref);
    for (r in p) h.call(p, r) && !d.hasOwnProperty(r) && (c[r] = p[r]);
    if (b && b.defaultProps) for (r in p = b.defaultProps, p) c[r] === void 0 && (c[r] = p[r]);
    return { $$typeof: m, type: b, key: a, ref: u, props: c, _owner: n.current };
  }
  return q.Fragment = i, q.jsx = o, q.jsxs = o, q;
}
var W = {};
/**
 * @license React
 * react-jsx-runtime.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var $e;
function fs() {
  return $e || ($e = 1, process.env.NODE_ENV !== "production" && function() {
    var t = ie, m = Symbol.for("react.element"), i = Symbol.for("react.portal"), h = Symbol.for("react.fragment"), n = Symbol.for("react.strict_mode"), d = Symbol.for("react.profiler"), o = Symbol.for("react.provider"), b = Symbol.for("react.context"), p = Symbol.for("react.forward_ref"), s = Symbol.for("react.suspense"), r = Symbol.for("react.suspense_list"), c = Symbol.for("react.memo"), a = Symbol.for("react.lazy"), u = Symbol.for("react.offscreen"), j = Symbol.iterator, k = "@@iterator";
    function M(l) {
      if (l === null || typeof l != "object")
        return null;
      var f = j && l[j] || l[k];
      return typeof f == "function" ? f : null;
    }
    var $ = t.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;
    function A(l) {
      {
        for (var f = arguments.length, g = new Array(f > 1 ? f - 1 : 0), N = 1; N < f; N++)
          g[N - 1] = arguments[N];
        J("error", l, g);
      }
    }
    function J(l, f, g) {
      {
        var N = $.ReactDebugCurrentFrame, E = N.getStackAddendum();
        E !== "" && (f += "%s", g = g.concat([E]));
        var S = g.map(function(w) {
          return String(w);
        });
        S.unshift("Warning: " + f), Function.prototype.apply.call(console[l], console, S);
      }
    }
    var T = !1, ze = !1, Oe = !1, Me = !1, Le = !1, de;
    de = Symbol.for("react.module.reference");
    function Fe(l) {
      return !!(typeof l == "string" || typeof l == "function" || l === h || l === d || Le || l === n || l === s || l === r || Me || l === u || T || ze || Oe || typeof l == "object" && l !== null && (l.$$typeof === a || l.$$typeof === c || l.$$typeof === o || l.$$typeof === b || l.$$typeof === p || // This needs to include all possible module reference object
      // types supported by any Flight configuration anywhere since
      // we don't know which Flight build this will end up being used
      // with.
      l.$$typeof === de || l.getModuleId !== void 0));
    }
    function Ge(l, f, g) {
      var N = l.displayName;
      if (N)
        return N;
      var E = f.displayName || f.name || "";
      return E !== "" ? g + "(" + E + ")" : g;
    }
    function oe(l) {
      return l.displayName || "Context";
    }
    function L(l) {
      if (l == null)
        return null;
      if (typeof l.tag == "number" && A("Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."), typeof l == "function")
        return l.displayName || l.name || null;
      if (typeof l == "string")
        return l;
      switch (l) {
        case h:
          return "Fragment";
        case i:
          return "Portal";
        case d:
          return "Profiler";
        case n:
          return "StrictMode";
        case s:
          return "Suspense";
        case r:
          return "SuspenseList";
      }
      if (typeof l == "object")
        switch (l.$$typeof) {
          case b:
            var f = l;
            return oe(f) + ".Consumer";
          case o:
            var g = l;
            return oe(g._context) + ".Provider";
          case p:
            return Ge(l, l.render, "ForwardRef");
          case c:
            var N = l.displayName || null;
            return N !== null ? N : L(l.type) || "Memo";
          case a: {
            var E = l, S = E._payload, w = E._init;
            try {
              return L(w(S));
            } catch {
              return null;
            }
          }
        }
      return null;
    }
    var F = Object.assign, I = 0, ce, xe, me, ue, pe, he, be;
    function fe() {
    }
    fe.__reactDisabledLog = !0;
    function Ve() {
      {
        if (I === 0) {
          ce = console.log, xe = console.info, me = console.warn, ue = console.error, pe = console.group, he = console.groupCollapsed, be = console.groupEnd;
          var l = {
            configurable: !0,
            enumerable: !0,
            value: fe,
            writable: !0
          };
          Object.defineProperties(console, {
            info: l,
            log: l,
            warn: l,
            error: l,
            group: l,
            groupCollapsed: l,
            groupEnd: l
          });
        }
        I++;
      }
    }
    function Ue() {
      {
        if (I--, I === 0) {
          var l = {
            configurable: !0,
            enumerable: !0,
            writable: !0
          };
          Object.defineProperties(console, {
            log: F({}, l, {
              value: ce
            }),
            info: F({}, l, {
              value: xe
            }),
            warn: F({}, l, {
              value: me
            }),
            error: F({}, l, {
              value: ue
            }),
            group: F({}, l, {
              value: pe
            }),
            groupCollapsed: F({}, l, {
              value: he
            }),
            groupEnd: F({}, l, {
              value: be
            })
          });
        }
        I < 0 && A("disabledDepth fell below zero. This is a bug in React. Please file an issue.");
      }
    }
    var X = $.ReactCurrentDispatcher, Z;
    function Q(l, f, g) {
      {
        if (Z === void 0)
          try {
            throw Error();
          } catch (E) {
            var N = E.stack.trim().match(/\n( *(at )?)/);
            Z = N && N[1] || "";
          }
        return `
` + Z + l;
      }
    }
    var ee = !1, H;
    {
      var Ie = typeof WeakMap == "function" ? WeakMap : Map;
      H = new Ie();
    }
    function ge(l, f) {
      if (!l || ee)
        return "";
      {
        var g = H.get(l);
        if (g !== void 0)
          return g;
      }
      var N;
      ee = !0;
      var E = Error.prepareStackTrace;
      Error.prepareStackTrace = void 0;
      var S;
      S = X.current, X.current = null, Ve();
      try {
        if (f) {
          var w = function() {
            throw Error();
          };
          if (Object.defineProperty(w.prototype, "props", {
            set: function() {
              throw Error();
            }
          }), typeof Reflect == "object" && Reflect.construct) {
            try {
              Reflect.construct(w, []);
            } catch (P) {
              N = P;
            }
            Reflect.construct(l, [], w);
          } else {
            try {
              w.call();
            } catch (P) {
              N = P;
            }
            l.call(w.prototype);
          }
        } else {
          try {
            throw Error();
          } catch (P) {
            N = P;
          }
          l();
        }
      } catch (P) {
        if (P && N && typeof P.stack == "string") {
          for (var y = P.stack.split(`
`), D = N.stack.split(`
`), _ = y.length - 1, R = D.length - 1; _ >= 1 && R >= 0 && y[_] !== D[R]; )
            R--;
          for (; _ >= 1 && R >= 0; _--, R--)
            if (y[_] !== D[R]) {
              if (_ !== 1 || R !== 1)
                do
                  if (_--, R--, R < 0 || y[_] !== D[R]) {
                    var z = `
` + y[_].replace(" at new ", " at ");
                    return l.displayName && z.includes("<anonymous>") && (z = z.replace("<anonymous>", l.displayName)), typeof l == "function" && H.set(l, z), z;
                  }
                while (_ >= 1 && R >= 0);
              break;
            }
        }
      } finally {
        ee = !1, X.current = S, Ue(), Error.prepareStackTrace = E;
      }
      var U = l ? l.displayName || l.name : "", G = U ? Q(U) : "";
      return typeof l == "function" && H.set(l, G), G;
    }
    function Ke(l, f, g) {
      return ge(l, !1);
    }
    function qe(l) {
      var f = l.prototype;
      return !!(f && f.isReactComponent);
    }
    function B(l, f, g) {
      if (l == null)
        return "";
      if (typeof l == "function")
        return ge(l, qe(l));
      if (typeof l == "string")
        return Q(l);
      switch (l) {
        case s:
          return Q("Suspense");
        case r:
          return Q("SuspenseList");
      }
      if (typeof l == "object")
        switch (l.$$typeof) {
          case p:
            return Ke(l.render);
          case c:
            return B(l.type, f, g);
          case a: {
            var N = l, E = N._payload, S = N._init;
            try {
              return B(S(E), f, g);
            } catch {
            }
          }
        }
      return "";
    }
    var K = Object.prototype.hasOwnProperty, je = {}, ve = $.ReactDebugCurrentFrame;
    function Y(l) {
      if (l) {
        var f = l._owner, g = B(l.type, l._source, f ? f.type : null);
        ve.setExtraStackFrame(g);
      } else
        ve.setExtraStackFrame(null);
    }
    function We(l, f, g, N, E) {
      {
        var S = Function.call.bind(K);
        for (var w in l)
          if (S(l, w)) {
            var y = void 0;
            try {
              if (typeof l[w] != "function") {
                var D = Error((N || "React class") + ": " + g + " type `" + w + "` is invalid; it must be a function, usually from the `prop-types` package, but received `" + typeof l[w] + "`.This often happens because of typos such as `PropTypes.function` instead of `PropTypes.func`.");
                throw D.name = "Invariant Violation", D;
              }
              y = l[w](f, w, N, g, null, "SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED");
            } catch (_) {
              y = _;
            }
            y && !(y instanceof Error) && (Y(E), A("%s: type specification of %s `%s` is invalid; the type checker function must return `null` or an `Error` but returned a %s. You may have forgotten to pass an argument to the type checker creator (arrayOf, instanceOf, objectOf, oneOf, oneOfType, and shape all require an argument).", N || "React class", g, w, typeof y), Y(null)), y instanceof Error && !(y.message in je) && (je[y.message] = !0, Y(E), A("Failed %s type: %s", g, y.message), Y(null));
          }
      }
    }
    var Je = Array.isArray;
    function se(l) {
      return Je(l);
    }
    function Qe(l) {
      {
        var f = typeof Symbol == "function" && Symbol.toStringTag, g = f && l[Symbol.toStringTag] || l.constructor.name || "Object";
        return g;
      }
    }
    function He(l) {
      try {
        return Ne(l), !1;
      } catch {
        return !0;
      }
    }
    function Ne(l) {
      return "" + l;
    }
    function ye(l) {
      if (He(l))
        return A("The provided key is an unsupported type %s. This value must be coerced to a string before before using it here.", Qe(l)), Ne(l);
    }
    var we = $.ReactCurrentOwner, Be = {
      key: !0,
      ref: !0,
      __self: !0,
      __source: !0
    }, Ee, ke;
    function Ye(l) {
      if (K.call(l, "ref")) {
        var f = Object.getOwnPropertyDescriptor(l, "ref").get;
        if (f && f.isReactWarning)
          return !1;
      }
      return l.ref !== void 0;
    }
    function Xe(l) {
      if (K.call(l, "key")) {
        var f = Object.getOwnPropertyDescriptor(l, "key").get;
        if (f && f.isReactWarning)
          return !1;
      }
      return l.key !== void 0;
    }
    function Ze(l, f) {
      typeof l.ref == "string" && we.current;
    }
    function es(l, f) {
      {
        var g = function() {
          Ee || (Ee = !0, A("%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://reactjs.org/link/special-props)", f));
        };
        g.isReactWarning = !0, Object.defineProperty(l, "key", {
          get: g,
          configurable: !0
        });
      }
    }
    function ss(l, f) {
      {
        var g = function() {
          ke || (ke = !0, A("%s: `ref` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://reactjs.org/link/special-props)", f));
        };
        g.isReactWarning = !0, Object.defineProperty(l, "ref", {
          get: g,
          configurable: !0
        });
      }
    }
    var ts = function(l, f, g, N, E, S, w) {
      var y = {
        // This tag allows us to uniquely identify this as a React Element
        $$typeof: m,
        // Built-in properties that belong on the element
        type: l,
        key: f,
        ref: g,
        props: w,
        // Record the component responsible for creating this element.
        _owner: S
      };
      return y._store = {}, Object.defineProperty(y._store, "validated", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: !1
      }), Object.defineProperty(y, "_self", {
        configurable: !1,
        enumerable: !1,
        writable: !1,
        value: N
      }), Object.defineProperty(y, "_source", {
        configurable: !1,
        enumerable: !1,
        writable: !1,
        value: E
      }), Object.freeze && (Object.freeze(y.props), Object.freeze(y)), y;
    };
    function as(l, f, g, N, E) {
      {
        var S, w = {}, y = null, D = null;
        g !== void 0 && (ye(g), y = "" + g), Xe(f) && (ye(f.key), y = "" + f.key), Ye(f) && (D = f.ref, Ze(f, E));
        for (S in f)
          K.call(f, S) && !Be.hasOwnProperty(S) && (w[S] = f[S]);
        if (l && l.defaultProps) {
          var _ = l.defaultProps;
          for (S in _)
            w[S] === void 0 && (w[S] = _[S]);
        }
        if (y || D) {
          var R = typeof l == "function" ? l.displayName || l.name || "Unknown" : l;
          y && es(w, R), D && ss(w, R);
        }
        return ts(l, y, D, E, N, we.current, w);
      }
    }
    var te = $.ReactCurrentOwner, Se = $.ReactDebugCurrentFrame;
    function V(l) {
      if (l) {
        var f = l._owner, g = B(l.type, l._source, f ? f.type : null);
        Se.setExtraStackFrame(g);
      } else
        Se.setExtraStackFrame(null);
    }
    var ae;
    ae = !1;
    function le(l) {
      return typeof l == "object" && l !== null && l.$$typeof === m;
    }
    function Ce() {
      {
        if (te.current) {
          var l = L(te.current.type);
          if (l)
            return `

Check the render method of \`` + l + "`.";
        }
        return "";
      }
    }
    function ls(l) {
      return "";
    }
    var _e = {};
    function rs(l) {
      {
        var f = Ce();
        if (!f) {
          var g = typeof l == "string" ? l : l.displayName || l.name;
          g && (f = `

Check the top-level render call using <` + g + ">.");
        }
        return f;
      }
    }
    function Re(l, f) {
      {
        if (!l._store || l._store.validated || l.key != null)
          return;
        l._store.validated = !0;
        var g = rs(f);
        if (_e[g])
          return;
        _e[g] = !0;
        var N = "";
        l && l._owner && l._owner !== te.current && (N = " It was passed a child from " + L(l._owner.type) + "."), V(l), A('Each child in a list should have a unique "key" prop.%s%s See https://reactjs.org/link/warning-keys for more information.', g, N), V(null);
      }
    }
    function Ae(l, f) {
      {
        if (typeof l != "object")
          return;
        if (se(l))
          for (var g = 0; g < l.length; g++) {
            var N = l[g];
            le(N) && Re(N, f);
          }
        else if (le(l))
          l._store && (l._store.validated = !0);
        else if (l) {
          var E = M(l);
          if (typeof E == "function" && E !== l.entries)
            for (var S = E.call(l), w; !(w = S.next()).done; )
              le(w.value) && Re(w.value, f);
        }
      }
    }
    function ns(l) {
      {
        var f = l.type;
        if (f == null || typeof f == "string")
          return;
        var g;
        if (typeof f == "function")
          g = f.propTypes;
        else if (typeof f == "object" && (f.$$typeof === p || // Note: Memo only checks outer props here.
        // Inner props are checked in the reconciler.
        f.$$typeof === c))
          g = f.propTypes;
        else
          return;
        if (g) {
          var N = L(f);
          We(g, l.props, "prop", N, l);
        } else if (f.PropTypes !== void 0 && !ae) {
          ae = !0;
          var E = L(f);
          A("Component %s declared `PropTypes` instead of `propTypes`. Did you misspell the property assignment?", E || "Unknown");
        }
        typeof f.getDefaultProps == "function" && !f.getDefaultProps.isReactClassApproved && A("getDefaultProps is only used on classic React.createClass definitions. Use a static property named `defaultProps` instead.");
      }
    }
    function is(l) {
      {
        for (var f = Object.keys(l.props), g = 0; g < f.length; g++) {
          var N = f[g];
          if (N !== "children" && N !== "key") {
            V(l), A("Invalid prop `%s` supplied to `React.Fragment`. React.Fragment can only have `key` and `children` props.", N), V(null);
            break;
          }
        }
        l.ref !== null && (V(l), A("Invalid attribute `ref` supplied to `React.Fragment`."), V(null));
      }
    }
    var Te = {};
    function De(l, f, g, N, E, S) {
      {
        var w = Fe(l);
        if (!w) {
          var y = "";
          (l === void 0 || typeof l == "object" && l !== null && Object.keys(l).length === 0) && (y += " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports.");
          var D = ls();
          D ? y += D : y += Ce();
          var _;
          l === null ? _ = "null" : se(l) ? _ = "array" : l !== void 0 && l.$$typeof === m ? (_ = "<" + (L(l.type) || "Unknown") + " />", y = " Did you accidentally export a JSX literal instead of a component?") : _ = typeof l, A("React.jsx: type is invalid -- expected a string (for built-in components) or a class/function (for composite components) but got: %s.%s", _, y);
        }
        var R = as(l, f, g, E, S);
        if (R == null)
          return R;
        if (w) {
          var z = f.children;
          if (z !== void 0)
            if (N)
              if (se(z)) {
                for (var U = 0; U < z.length; U++)
                  Ae(z[U], l);
                Object.freeze && Object.freeze(z);
              } else
                A("React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead.");
            else
              Ae(z, l);
        }
        if (K.call(f, "key")) {
          var G = L(l), P = Object.keys(f).filter(function(us) {
            return us !== "key";
          }), re = P.length > 0 ? "{key: someKey, " + P.join(": ..., ") + ": ...}" : "{key: someKey}";
          if (!Te[G + re]) {
            var ms = P.length > 0 ? "{" + P.join(": ..., ") + ": ...}" : "{}";
            A(`A props object containing a "key" prop is being spread into JSX:
  let props = %s;
  <%s {...props} />
React keys must be passed directly to JSX without using spread:
  let props = %s;
  <%s key={someKey} {...props} />`, re, G, ms, G), Te[G + re] = !0;
          }
        }
        return l === h ? is(R) : ns(R), R;
      }
    }
    function ds(l, f, g) {
      return De(l, f, g, !0);
    }
    function os(l, f, g) {
      return De(l, f, g, !1);
    }
    var cs = os, xs = ds;
    W.Fragment = h, W.jsx = cs, W.jsxs = xs;
  }()), W;
}
process.env.NODE_ENV === "production" ? ne.exports = bs() : ne.exports = fs();
var e = ne.exports;
const js = ({
  nodes: t,
  edges: m,
  height: i = 600,
  autoGenerate: h = !0
}) => {
  const d = v.getState().features.pygraphistryEnabled, [o, b] = x(null), [p, s] = x(!1), [r, c] = x(null), a = async () => {
    if (d) {
      s(!0), c(null);
      try {
        const u = await v.generateVisualization({
          nodes: t,
          edges: m,
          layout: "force_directed"
        });
        u ? b(u) : c("Failed to generate visualization URL. Ensure backend is running and configured.");
      } catch (u) {
        c(u instanceof Error ? u.message : "Unknown error");
      } finally {
        s(!1);
      }
    }
  };
  return C(() => {
    d && h && t.length > 0 && a();
  }, [t, m, h, d]), d ? p ? /* @__PURE__ */ e.jsx("div", { className: "p-4 text-center", children: "Loading PyGraphistry Visualization..." }) : r ? /* @__PURE__ */ e.jsxs("div", { className: "p-4 text-red-500 border border-red-200 rounded", children: [
    "Error: ",
    r
  ] }) : o ? /* @__PURE__ */ e.jsxs("div", { className: "relative w-full overflow-hidden rounded-lg shadow-lg border border-gray-200", children: [
    /* @__PURE__ */ e.jsx(
      "iframe",
      {
        src: o,
        width: "100%",
        height: i,
        frameBorder: "0",
        title: "PyGraphistry Visualization",
        scrolling: "no",
        allowFullScreen: !0
      }
    ),
    /* @__PURE__ */ e.jsxs("div", { className: "absolute top-2 right-2 flex space-x-2", children: [
      /* @__PURE__ */ e.jsx(
        "a",
        {
          href: o,
          target: "_blank",
          rel: "noopener noreferrer",
          className: "p-2 bg-white/80 backdrop-blur rounded hover:bg-white text-xs font-medium",
          children: "Open External"
        }
      ),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: a,
          className: "p-2 bg-white/80 backdrop-blur rounded hover:bg-white text-xs font-medium",
          children: "Refresh"
        }
      )
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-4 text-center", children: /* @__PURE__ */ e.jsx(
    "button",
    {
      onClick: a,
      className: "px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700",
      children: "Generate Graphistry Visualization"
    }
  ) }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-gray-200 rounded-lg bg-gray-50 text-gray-500", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "PyGraphistry visualization is currently disabled in settings." }) });
}, vs = ({
  data: t,
  variableNames: m,
  height: i = 500
}) => {
  const n = v.getState().features.causalDiscoveryEnabled, [d, o] = x(!1), [b, p] = x(null), [s, r] = x(null), c = async () => {
    if (n) {
      o(!0), r(null);
      try {
        const a = await fetch("/api/openevolve/causal/discover", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            data: t,
            variable_names: m,
            method: "pc"
          })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Causal discovery failed");
        }
        const u = await a.json();
        p(u);
      } catch (a) {
        r(a instanceof Error ? a.message : "Unknown error");
      } finally {
        o(!1);
      }
    }
  };
  return n ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-semibold text-gray-800", children: "Causal Discovery Analysis" }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: d,
          className: "px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors",
          children: d ? "Analyzing..." : "Run Causal Discovery"
        }
      )
    ] }),
    s && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded", children: s }),
    !b && !d && /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col items-center justify-center h-64 border-2 border-dashed border-gray-200 rounded-lg text-gray-400", children: [
      /* @__PURE__ */ e.jsx("p", { children: 'Click "Run Causal Discovery" to analyze variables:' }),
      /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2 mt-2", children: m.map((a) => /* @__PURE__ */ e.jsx("span", { className: "px-2 py-1 bg-gray-100 rounded text-xs", children: a }, a)) })
    ] }),
    b && /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "border rounded p-4 bg-gray-50", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "font-medium mb-2", children: "Discovered Relationships" }),
        /* @__PURE__ */ e.jsx("ul", { className: "space-y-1", children: b.edges.map((a, u) => /* @__PURE__ */ e.jsxs("li", { className: "text-sm flex items-center", children: [
          /* @__PURE__ */ e.jsx("span", { className: "font-bold text-indigo-600", children: b.nodes[a[0]] }),
          /* @__PURE__ */ e.jsx("span", { className: "mx-2", children: "→" }),
          /* @__PURE__ */ e.jsx("span", { className: "font-bold text-teal-600", children: b.nodes[a[1]] })
        ] }, u)) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "border rounded p-4 bg-gray-50", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "font-medium mb-2", children: "Algorithm Insights" }),
        /* @__PURE__ */ e.jsxs("p", { className: "text-sm text-gray-600", children: [
          "Method: ",
          /* @__PURE__ */ e.jsx("span", { className: "font-mono bg-white px-1 border rounded", children: b.algorithm })
        ] }),
        /* @__PURE__ */ e.jsxs("p", { className: "text-sm text-gray-600 mt-2", children: [
          "Detected ",
          b.edges.length,
          " causal pathways across ",
          b.nodes.length,
          " variables."
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-indigo-100 rounded-lg bg-indigo-50/30 text-indigo-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Causal Discovery visualization is currently disabled in settings." }) });
}, Ns = ({
  problemType: t,
  initialValue: m = 10
}) => {
  const h = v.getState().features.optimizationEnabled, [n, d] = x(!1), [o, b] = x([]), [p, s] = x(null), r = async () => {
    if (h) {
      d(!0);
      try {
        const c = await fetch("/api/openevolve/optimization/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            problem_type: t,
            initial_value: m
          })
        });
        if (!c.ok)
          throw new Error("Optimization failed");
        const a = await c.json();
        b(a.convergence || []), s(a.optimal_value);
      } catch (c) {
        console.error("Optimization failed:", c);
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-lg bg-slate-50 space-y-4", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "NeuroMANCER Optimization" }),
        /* @__PURE__ */ e.jsxs("p", { className: "text-xs text-slate-500", children: [
          "Problem: ",
          t
        ] })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: r,
          disabled: n,
          className: "px-4 py-2 bg-emerald-600 text-white rounded shadow hover:bg-emerald-700 disabled:opacity-50",
          children: n ? "Solving..." : "Run Optimization"
        }
      )
    ] }),
    p !== null && /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-white rounded border border-emerald-100 flex justify-around", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "text-center", children: [
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-400 uppercase", children: "Optimal Value" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono text-emerald-600", children: p.toFixed(4) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "text-center", children: [
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-400 uppercase", children: "Iterations" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono text-slate-700", children: o.length })
      ] })
    ] }),
    o.length > 0 && /* @__PURE__ */ e.jsx("div", { className: "h-40 flex items-end space-x-1 border-b border-l p-2 bg-white", children: o.map((c, a) => /* @__PURE__ */ e.jsx(
      "div",
      {
        className: "bg-emerald-400 w-full hover:bg-emerald-500 transition-all",
        style: { height: `${c / m * 100}%` },
        title: `Step ${a}: ${c.toFixed(2)}`
      },
      a
    )) }),
    /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 text-center italic", children: "NeuroMANCER: Differentiable Programming with Physics Constraints" })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "NeuroMANCER Optimization visualization is currently disabled in settings." }) });
}, ys = () => {
  const [t, m] = x(v.getState()), i = (n) => {
    const d = { ...t.features, [n]: !t.features[n] };
    v.updateFeatures(d), m(v.getState());
  }, h = (n) => {
    const { name: d, value: o } = n.target;
    v.updateConfig({ [d]: o }), m(v.getState());
  };
  return /* @__PURE__ */ e.jsxs("div", { className: "p-6 bg-white rounded-xl shadow-sm border border-slate-200 space-y-6", children: [
    /* @__PURE__ */ e.jsxs("div", { children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Visualization Settings" }),
      /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-500 text-balance", children: "Manage visualization components and their respective backend configurations." })
    ] }),
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400", children: "Feature Toggles" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "PyGraphistry" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Interactive Knowledge Graphs" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("pygraphistryEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.pygraphistryEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.pygraphistryEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Causal Discovery" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Identify Causal Mechanisms" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("causalDiscoveryEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.causalDiscoveryEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.causalDiscoveryEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "NeuroMANCER" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Loss Landscape Visualization" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("optimizationEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.optimizationEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.optimizationEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Uncertainty Quantification" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Sensitivity Analysis (uqtestfuns)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("uqEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.uqEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.uqEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Chemical Knowledge" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Molecular Explorer (GlobalChem)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("globalChemEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.globalChemEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.globalChemEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Scientific Experimentation" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Automated Protocols (Curie)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("curieEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.curieEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.curieEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Temporal Knowledge Graph" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Time-aware Facts (Graphiti)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("temporalGraphEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.temporalGraphEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.temporalGraphEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Knowledge Extraction" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Schema-guided NER/RE (OneKE)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("onekeEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.onekeEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.onekeEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Autoformalization" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Math to Lean4 Proofs (LeanAide)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("leanAideEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.leanAideEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.leanAideEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "SOP Generation" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Structured Operating Procedures" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("sopEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.sopEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.sopEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Adversarial Validation" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Red/Blue Team Robustness" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("adversarialEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.adversarialEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.adversarialEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Pattern Mining" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Frequent Itemset Discovery (PAMI)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("pamiEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.pamiEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.pamiEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Context Analytics" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Team & Gauntlet Performance (ACE)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("aceEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.aceEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.aceEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Recursive Meta-Agents" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Hierarchical Orchestration (ROMA)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("romaEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.romaEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.romaEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Multi-Agent Data Processing" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Blue/Red/Gold Workflow (DataPizza)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("datapizzaEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.datapizzaEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.datapizzaEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Active Reliability" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Deterministic Verification (ACE + Steer)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("steerEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.steerEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.steerEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Project Management" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Ticket Tracking (CrewAI)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("crewaiEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.crewaiEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.crewaiEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Autonomous Development" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Task Decomposition (Claudiomiro)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("claudiomiroEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.claudiomiroEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.claudiomiroEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Research Methodology" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "8-Stage Lifecycle (Research-Quest)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("researchQuestEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.researchQuestEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.researchQuestEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Knowledge Graph Generation" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Text to KG (KG-GEN)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("kgEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.kgEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.kgEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Sovereign Workflow Monitoring" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Real-time SGD Metrics (Advanced Monitoring)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("sgdEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.sgdEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.sgdEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Global Performance Analytics" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Cross-Project Cost & Token Tracking" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("globalAnalyticsEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.globalAnalyticsEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.globalAnalyticsEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Quality-Diversity (MAP-Elites)" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Feature Space Optimization" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("mapElitesEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.mapElitesEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.mapElitesEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Mathematical Verification" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Algorithmic Correctness Analysis" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("verificationEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.verificationEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.verificationEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Problem Analysis" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Semantic Problem Decomposition" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("problemAnalysisEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.problemAnalysisEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.problemAnalysisEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Dependency Mapping" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Sub-problem DAG Visualization" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("dependencyEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.dependencyEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.dependencyEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Knowledge Artifact Mapping" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Relationship Graph (Artifact Manager)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("artifactGraphEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.artifactGraphEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.artifactGraphEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Symbolic Logic Constraints" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Formal Logic & Lean4 Verification (SCE)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("sceEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.sceEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.sceEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Static Code Analysis" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Security & Quality Scanning (DeepStatic)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("staticAnalysisEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.staticAnalysisEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.staticAnalysisEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Logic-to-Loss Translation" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Differentiable Constraints (LLTL)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("lltlEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.lltlEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.lltlEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Multi-Agent Collaboration" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Real-time Session Sync & Conflict Resolution" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("collaborationEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.collaborationEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.collaborationEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Workflow Execution Monitor" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Real-time Pipeline & Resource Tracking" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("workflowMonitorEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.workflowMonitorEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.workflowMonitorEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Evolution Ancestry & Lineage" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Parent-Child Improvement Tracing" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("lineageEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.lineageEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.lineageEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Gauntlet Effectiveness" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Validation Catch-Rate Analysis" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("gauntletEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.gauntletEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.gauntletEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Knowledge Pattern Discovery" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "ML-based Solution Clustering (Miner)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("patternMiningEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.patternMiningEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.patternMiningEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Dynamic Gauntlet Adaptation" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Active Validation Strictness Optimization" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("adaptationEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.adaptationEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.adaptationEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "High-Performance Logic Audit" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "O(n log n) Contradiction Detection (DITO)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("ditoEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.ditoEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.ditoEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Knowledge Retrieval (RAG)" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Augmented Context Recovery (Ragbits)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("ragEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.ragEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.ragEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Knowledge Extraction" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Structured Entity & Relation Discovery (DeepKE)" })
        ] }),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: () => i("deepkeEnabled"),
            className: `w-12 h-6 rounded-full transition-colors relative ${t.features.deepkeEnabled ? "bg-indigo-600" : "bg-slate-300"}`,
            children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${t.features.deepkeEnabled ? "translate-x-6" : ""}` })
          }
        )
      ] }),
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6", children: "Decision & Search Engines" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Multi-Agent Voting" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Consensus & Proposals (MAKER)" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("makerEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.makerEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.makerEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Multi-Dim Processing" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Dimensional Synthesis (MDAP)" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("mdapEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.mdapEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.mdapEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Monte Carlo Tree Search" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Decision Space Optimization (MCTS)" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("mctsEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.mctsEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.mctsEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Hybrid MCTS" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Evolution + Search Synergy" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("hybridMCTSEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.hybridMCTSEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.hybridMCTSEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6", children: "Quality & Reliability" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Multi-Judge Evaluator" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Consensus Scoring (Evaluator Team)" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("evaluatorTeamEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.evaluatorTeamEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.evaluatorTeamEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Red Team Security" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Autonomous Vulnerability Probing" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("redTeamEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.redTeamEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.redTeamEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Blue Team Defense" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Security Hardening & Shielding" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("blueTeamEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.blueTeamEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.blueTeamEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "QA Suite" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Comprehensive Test Coverage" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("qaSuiteEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.qaSuiteEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.qaSuiteEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "RESE Reliability" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "System-wide Fault Tolerance" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("reseEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.reseEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.reseEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6", children: "Scientific & Discovery" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Material KG" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Materials Science Knowledge Graph" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("materialKGEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.materialKGEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.materialKGEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "GNoME Discovery" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "AI-driven Material Exploration" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("gnomeEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.gnomeEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.gnomeEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Physics-NeMo" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "High-fidelity Simulation" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("physicsNemoEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.physicsNemoEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.physicsNemoEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "PINNs Library" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Physics-Informed Neural Networks" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("pinnsEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.pinnsEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.pinnsEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "PyLabRobot" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Laboratory Robotics Automation" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("pylabrobotEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.pylabrobotEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.pylabrobotEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6", children: "Graph ML & Embeddings" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "KarateClub" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Unsupervised Graph ML (DeepWalk/node2vec)" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("karateclubEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.karateclubEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.karateclubEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "NeuralKG" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Knowledge Graph Embedding Framework" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("neuralKGEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.neuralKGEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.neuralKGEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6", children: "Roadmap Agents (Category 9)" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "AutoGPT Swarms" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Autonomous Task Loops" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("autogptEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.autogptEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.autogptEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Microsoft AutoGen" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Multi-agent Conversation Dynamics" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("autogenEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.autogenEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.autogenEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "MetaGPT Firm" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Software Company Simulation" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("metagptEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.metagptEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.metagptEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "AI Scientist" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Automated Scientific Hypothesizing" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("aiScientistEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.aiScientistEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.aiScientistEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400 mt-6", children: "Error Analysis & Gap Filling" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Uncertainty Analysis" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Propagation & Sensitivity (Uncertainpy)" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("uncertainpyEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.uncertainpyEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.uncertainpyEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "LLM Risk Analyzer" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Vulnerability & Bias Detection" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("riskAnalyzerEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.riskAnalyzerEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.riskAnalyzerEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "SOP Optimization" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Procedure Enhancement (LLM4IAS)" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("llm4iasEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.llm4iasEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.llm4iasEnabled ? "translate-x-6" : ""}` }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "font-medium text-slate-700", children: "Integration Assessment" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "ClaraVerse Compatibility Auditing" })
        ] }),
        /* @__PURE__ */ e.jsx("button", { onClick: () => i("claraverseEnabled"), className: `w-12 h-6 rounded-full relative ${t.features.claraverseEnabled ? "bg-indigo-600" : "bg-slate-300"}`, children: /* @__PURE__ */ e.jsx("span", { className: `absolute top-1 left-1 bg-white w-4 h-4 rounded-full ${t.features.claraverseEnabled ? "translate-x-6" : ""}` }) })
      ] })
    ] }),
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
      /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-semibold uppercase tracking-wider text-slate-400", children: "Configuration" }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
        /* @__PURE__ */ e.jsx("label", { className: "text-sm font-medium text-slate-600 block", children: "Graphistry API Key" }),
        /* @__PURE__ */ e.jsx(
          "input",
          {
            type: "password",
            name: "apiKey",
            value: t.config.apiKey || "",
            onChange: h,
            placeholder: "sk_...",
            className: "w-full p-2 border rounded-md text-sm font-mono focus:ring-2 focus:ring-indigo-500 outline-none"
          }
        )
      ] })
    ] })
  ] });
}, ws = ({
  testFunction: t,
  nSamples: m = 500
}) => {
  const h = v.getState().features.uqEnabled, [n, d] = x(!1), [o, b] = x(null), [p, s] = x(null), r = async () => {
    if (h) {
      d(!0), s(null);
      try {
        const c = await fetch("/api/openevolve/uq/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            function_name: t.toLowerCase(),
            n_samples: m
          })
        });
        if (!c.ok) {
          const u = await c.json();
          throw new Error(u.detail || "UQ analysis failed");
        }
        const a = await c.json();
        b(a);
      } catch (c) {
        s(c instanceof Error ? c.message : "Unknown error");
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsxs("h3", { className: "text-lg font-semibold text-gray-800", children: [
          "UQ Analysis: ",
          t
        ] }),
        /* @__PURE__ */ e.jsxs("p", { className: "text-xs text-gray-500", children: [
          m,
          " Monte Carlo samples"
        ] })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: r,
          disabled: n,
          className: "px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors",
          children: n ? "Analyzing..." : "Run UQ Pipeline"
        }
      )
    ] }),
    p && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded", children: p }),
    o && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6", children: [
      /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4", children: [
        { label: "Mean", value: o.statistics.mean },
        { label: "Std Dev", value: o.statistics.std },
        { label: "Min", value: o.statistics.min },
        { label: "Max", value: o.statistics.max }
      ].map((c) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-md border text-center", children: [
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] uppercase text-slate-400 font-bold", children: c.label }),
        /* @__PURE__ */ e.jsx("p", { className: "text-lg font-mono text-slate-700", children: c.value.toFixed(4) })
      ] }, c.label)) }),
      o.sensitivity && /* @__PURE__ */ e.jsxs("div", { className: "border rounded-lg p-4 bg-gray-50", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-600 mb-3 uppercase tracking-tight", children: "Sobol Sensitivity Indices" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-3", children: o.sensitivity.first_order.map((c, a) => /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-xs font-medium", children: [
            /* @__PURE__ */ e.jsxs("span", { children: [
              "Input X",
              a + 1
            ] }),
            /* @__PURE__ */ e.jsxs("span", { children: [
              (c * 100).toFixed(1),
              "%"
            ] })
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: "w-full bg-gray-200 rounded-full h-1.5", children: /* @__PURE__ */ e.jsx(
            "div",
            {
              className: "bg-blue-500 h-1.5 rounded-full",
              style: { width: `${c * 100}%` }
            }
          ) })
        ] }, a)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-blue-100 rounded-lg bg-blue-50/30 text-blue-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Uncertainty Quantification is currently disabled in settings." }) });
}, Es = ({
  initialList: t = "fda_approved"
}) => {
  const i = v.getState().features.globalChemEnabled, [h, n] = x(!1), [d, o] = x([]), [b, p] = x(null), [s, r] = x(""), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/chem/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: s,
            list_name: t
          })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Chemical search failed");
        }
        const u = await a.json();
        o(u.chemicals);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col md:flex-row md:items-end gap-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex-1 space-y-1", children: [
        /* @__PURE__ */ e.jsx("label", { className: "text-xs font-bold text-slate-400 uppercase", children: "Search Molecules" }),
        /* @__PURE__ */ e.jsx(
          "input",
          {
            type: "text",
            value: s,
            onChange: (a) => r(a.target.value),
            onKeyDown: (a) => a.key === "Enter" && c(),
            placeholder: "Search by name or SMILES...",
            className: "w-full p-2 border rounded-md focus:ring-2 focus:ring-teal-500 outline-none"
          }
        )
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h,
          className: "px-6 py-2 bg-teal-600 text-white rounded hover:bg-teal-700 disabled:opacity-50 transition-colors h-[42px]",
          children: h ? "Searching..." : "Search"
        }
      )
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4", children: [
      d.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-lg hover:border-teal-300 transition-colors bg-slate-50 group", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start mb-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "font-bold text-slate-800 truncate", title: a.name, children: a.name }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] px-1.5 py-0.5 bg-teal-100 text-teal-700 rounded font-medium", children: a.list || "GlobalChem" })
        ] }),
        /* @__PURE__ */ e.jsx("div", { className: "bg-white p-2 rounded border mb-2 overflow-hidden h-32 flex items-center justify-center", children: /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-mono text-slate-400 break-all leading-tight", children: a.smiles }) }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mt-auto", children: [
          /* @__PURE__ */ e.jsx("button", { className: "text-[10px] text-teal-600 font-bold uppercase group-hover:underline", children: "View Details" }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] text-slate-400 font-mono", children: a.molecular_weight ? `${a.molecular_weight.toFixed(1)} g/mol` : "" })
        ] })
      ] }, u)),
      !h && d.length === 0 && !b && /* @__PURE__ */ e.jsx("div", { className: "col-span-full py-12 text-center text-slate-400 border-2 border-dashed rounded-lg", children: /* @__PURE__ */ e.jsx("p", { children: 'No chemical data loaded. Try searching for "Aspirin" or "Caffeine".' }) })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-teal-100 rounded-lg bg-teal-50/30 text-teal-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Chemical Knowledge Explorer is currently disabled in settings." }) });
}, ks = ({
  initialHypothesis: t = "",
  domain: m = "physics"
}) => {
  const h = v.getState().features.curieEnabled, [n, d] = x(!1), [o, b] = x(null), [p, s] = x(null), [r, c] = x(t), a = async () => {
    if (h) {
      d(!0), s(null);
      try {
        const u = await fetch("/api/openevolve/curie/design", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            hypothesis: r,
            domain: m
          })
        });
        if (!u.ok) {
          const k = await u.json();
          throw new Error(k.detail || "Experiment design failed");
        }
        const j = await u.json();
        b(j);
      } catch (u) {
        s(u instanceof Error ? u.message : "Unknown error");
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-semibold text-gray-800", children: "Scientific Experiment Designer" }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: r,
          onChange: (u) => c(u.target.value),
          placeholder: "Enter your scientific hypothesis here...",
          className: "w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-amber-500 outline-none text-sm"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: a,
          disabled: n || !r,
          className: "px-6 py-2 bg-amber-600 text-white rounded hover:bg-amber-700 disabled:opacity-50 transition-colors",
          children: n ? "Designing Protocol..." : "Design Experiment"
        }
      ) })
    ] }),
    p && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: p }),
    o && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-amber-50 rounded-lg border border-amber-100", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-2", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-amber-700 uppercase tracking-widest", children: "Protocol Generated" }),
          /* @__PURE__ */ e.jsx("span", { className: "text-xs font-mono text-amber-600", children: o.protocol_id })
        ] }),
        /* @__PURE__ */ e.jsx("h4", { className: "font-bold text-slate-800", children: "Hypothesis Components" }),
        /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-4 mt-2", children: [
          /* @__PURE__ */ e.jsxs("div", { children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 uppercase font-bold", children: "Independent" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-700", children: o.hypothesis.independent_variables.join(", ") || "N/A" })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 uppercase font-bold", children: "Dependent" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-700", children: o.hypothesis.dependent_variables.join(", ") || "N/A" })
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-600 uppercase tracking-tight", children: "Execution Workflow" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: o.steps.map((u, j) => /* @__PURE__ */ e.jsxs("div", { className: "flex gap-3 items-start", children: [
          /* @__PURE__ */ e.jsx("div", { className: "flex-none w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center text-xs font-bold text-slate-500 border", children: j + 1 }),
          /* @__PURE__ */ e.jsx("div", { className: "flex-1 p-2 bg-white border rounded shadow-sm text-sm text-slate-700", children: u.description })
        ] }, j)) })
      ] }),
      o.equipment.length > 0 && /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2", children: o.equipment.map((u) => /* @__PURE__ */ e.jsxs("span", { className: "px-2 py-1 bg-slate-100 text-slate-600 rounded text-[10px] font-medium border", children: [
        "🛠️ ",
        u
      ] }, u)) })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-amber-100 rounded-lg bg-amber-50/30 text-amber-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Curie Experimentation is currently disabled in settings." }) });
}, Ss = ({
  initialQuery: t = ""
}) => {
  const i = v.getState().features.temporalGraphEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/graphiti/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: s })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Temporal search failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex gap-2", children: [
      /* @__PURE__ */ e.jsx(
        "input",
        {
          type: "text",
          value: s,
          onChange: (a) => r(a.target.value),
          onKeyDown: (a) => a.key === "Enter" && c(),
          placeholder: "Search temporal facts...",
          className: "flex-1 p-2 border rounded-md focus:ring-2 focus:ring-purple-500 outline-none text-sm"
        }
      ),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h || !s,
          className: "px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 transition-colors text-sm font-medium",
          children: h ? "Searching..." : "Search"
        }
      )
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center px-1", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold text-slate-400 uppercase", children: "Discovered Facts" }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-400", children: [
          d.edges.length,
          " results"
        ] })
      ] }),
      /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 gap-3", children: d.edges.map((a, u) => {
        var j, k;
        return /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors border-l-4 border-l-purple-400", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start mb-1", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-medium text-slate-800", children: a.fact }),
            a.valid_at && /* @__PURE__ */ e.jsx("span", { className: "text-[10px] bg-white px-1.5 py-0.5 rounded border text-slate-500 font-mono", children: new Date(a.valid_at).toLocaleDateString() })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "flex gap-2 items-center mt-2", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded-full font-bold", children: ((j = d.nodes.find((M) => M.uuid === a.source_node)) == null ? void 0 : j.name) || "Unknown" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-slate-300 text-[10px]", children: "→" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] px-1.5 py-0.5 bg-indigo-100 text-indigo-700 rounded-full font-bold", children: ((k = d.nodes.find((M) => M.uuid === a.target_node)) == null ? void 0 : k.name) || "Unknown" })
          ] })
        ] }, u);
      }) }),
      d.edges.length === 0 && /* @__PURE__ */ e.jsxs("div", { className: "py-12 text-center text-slate-400 border-2 border-dashed rounded-lg", children: [
        'No facts found for "',
        s,
        '".'
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-purple-100 rounded-lg bg-purple-50/30 text-purple-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Temporal Knowledge Graph is currently disabled in settings." }) });
}, Cs = ({
  initialText: t = ""
}) => {
  const i = v.getState().features.onekeEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/oneke/extract", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: s })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Extraction failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-semibold text-gray-800", children: "Schema-Guided Knowledge Extraction" }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: s,
          onChange: (a) => r(a.target.value),
          placeholder: "Paste text to extract structured knowledge...",
          className: "w-full p-3 border rounded-md min-h-[100px] focus:ring-2 focus:ring-orange-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h || !s,
          className: "px-6 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 disabled:opacity-50 transition-colors font-medium",
          children: h ? "Extracting..." : "Extract Entities & Relations"
        }
      ) })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Detected Entities" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex flex-wrap gap-2", children: [
          d.entities.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "px-3 py-1.5 bg-orange-50 border border-orange-100 rounded-full flex items-center gap-2 shadow-sm", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-sm font-semibold text-orange-800", children: a.text }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] bg-orange-200 text-orange-700 px-1.5 rounded-md font-bold uppercase", children: a.type })
          ] }, u)),
          d.entities.length === 0 && /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-400 italic", children: "No entities found." })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Extracted Relations" }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          d.relations.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 border rounded-lg flex items-center justify-between group hover:bg-white hover:border-orange-200 transition-all shadow-sm", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-sm font-bold text-slate-700", children: a.subject }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-mono text-orange-500 font-bold uppercase bg-white border px-2 py-0.5 rounded-full mx-2", children: a.predicate }),
            /* @__PURE__ */ e.jsx("span", { className: "text-sm font-bold text-slate-700", children: a.object })
          ] }, u)),
          d.relations.length === 0 && /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-400 italic", children: "No relations found." })
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-orange-100 rounded-lg bg-orange-50/30 text-orange-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Knowledge Extraction is currently disabled in settings." }) });
}, _s = ({
  initialTheorem: t = ""
}) => {
  const i = v.getState().features.leanAideEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/leanaide/formalize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ theorem_text: s })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Formalization failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-semibold text-gray-800", children: "Mathematical Autoformalization" }),
      /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Translate natural language math to Lean4 formal proofs." }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: s,
          onChange: (a) => r(a.target.value),
          placeholder: "Enter a mathematical theorem (e.g., 'There are infinitely many primes')...",
          className: "w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-emerald-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h || !s,
          className: "px-6 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 transition-colors font-medium",
          children: h ? "Formalizing..." : "Formalize to Lean4"
        }
      ) })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-900 rounded-lg border border-slate-800 overflow-x-auto", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-2 border-b border-slate-700 pb-2", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-emerald-400 uppercase tracking-widest", children: "Lean4 Output" }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-500 font-mono", children: [
            "Confidence: ",
            (d.confidence * 100).toFixed(0),
            "%"
          ] })
        ] }),
        /* @__PURE__ */ e.jsx("pre", { className: "text-xs font-mono text-slate-200 leading-relaxed", children: d.theorem_lean })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2 px-1", children: [
        /* @__PURE__ */ e.jsx("div", { className: `w-2 h-2 rounded-full ${d.proof_status === "verified" ? "bg-emerald-500" : "bg-amber-500"}` }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-xs font-medium text-slate-600 uppercase tracking-tight", children: [
          "Status: ",
          d.proof_status
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "LeanAide Autoformalization is currently disabled in settings." }) });
}, Rs = ({
  initialRequirement: t = "",
  domain: m = "general"
}) => {
  const h = v.getState().features.sopEnabled, [n, d] = x(!1), [o, b] = x(null), [p, s] = x(null), [r, c] = x(t), a = async () => {
    if (h) {
      d(!0), s(null);
      try {
        const u = await fetch("/api/openevolve/sop/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ requirement: r, domain: m })
        });
        if (!u.ok) {
          const k = await u.json();
          throw new Error(k.detail || "SOP generation failed");
        }
        const j = await u.json();
        b(j);
      } catch (u) {
        s(u instanceof Error ? u.message : "Unknown error");
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-semibold text-gray-800", children: "SOP Generator & Refiner" }),
      /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Generate turnkey-ready operating procedures from high-level goals." }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: r,
          onChange: (u) => c(u.target.value),
          placeholder: "Describe the process or requirement (e.g., 'Protocol for high-speed centrifugation of plasma samples')...",
          className: "w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-slate-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: a,
          disabled: n || !r,
          className: "px-6 py-2 bg-slate-800 text-white rounded hover:bg-slate-900 disabled:opacity-50 transition-colors font-medium shadow-sm",
          children: n ? "Generating SOP..." : "Generate Protocol"
        }
      ) })
    ] }),
    p && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: p }),
    o && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-between items-start", children: /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h2", { className: "text-xl font-bold text-slate-900", children: o.title }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex gap-2 mt-1", children: [
          /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] bg-slate-100 px-2 py-0.5 rounded border font-bold uppercase tracking-wider text-slate-600", children: [
            "v",
            o.version
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] bg-blue-50 px-2 py-0.5 rounded border border-blue-100 font-bold uppercase tracking-wider text-blue-600", children: o.status })
        ] })
      ] }) }),
      /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-600 leading-relaxed italic border-l-4 border-slate-200 pl-3", children: o.description }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Required Equipment" }),
          /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2", children: o.equipment.map((u, j) => /* @__PURE__ */ e.jsxs("span", { className: "px-2 py-1 bg-slate-50 border rounded text-xs font-medium text-slate-700", children: [
            "🛠️ ",
            typeof u == "string" ? u : u.name || "Unknown Device"
          ] }, j)) })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Execution Steps" }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: o.protocols.map((u) => /* @__PURE__ */ e.jsxs("div", { className: "flex gap-3 items-start p-2 bg-slate-50/50 rounded border border-slate-100", children: [
            /* @__PURE__ */ e.jsx("span", { className: "flex-none w-5 h-5 rounded-full bg-slate-800 text-white flex items-center justify-center text-[10px] font-bold", children: u.step_number }),
            /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700 font-medium", children: u.action })
          ] }, u.step_number)) })
        ] })
      ] }),
      (o.safety_protocols || o.quality_control) && /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-amber-50/50 rounded-lg border border-amber-100", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-amber-800 uppercase tracking-widest mb-2", children: "Safety & Quality Assurance" }),
        /* @__PURE__ */ e.jsxs("ul", { className: "list-disc list-inside space-y-1", children: [
          (o.safety_protocols || []).map((u, j) => /* @__PURE__ */ e.jsx("li", { className: "text-xs text-amber-900/80", children: u }, `s-${j}`)),
          (o.quality_control || []).map((u, j) => /* @__PURE__ */ e.jsx("li", { className: "text-xs text-slate-700", children: u }, `q-${j}`))
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "SOP Generation is currently disabled in settings." }) });
}, As = ({
  content: t,
  theorem: m = ""
}) => {
  const h = v.getState().features.adversarialEnabled, [n, d] = x(!1), [o, b] = x(null), [p, s] = x(null), r = async () => {
    if (h) {
      d(!0), s(null);
      try {
        const c = await fetch("/api/openevolve/adversarial/validate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: t, theorem: m })
        });
        if (!c.ok) {
          const u = await c.json();
          throw new Error(u.detail || "Adversarial validation failed");
        }
        const a = await c.json();
        b(a);
      } catch (c) {
        s(c instanceof Error ? c.message : "Unknown error");
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-semibold text-gray-800", children: "Red/Blue Team Validation" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Adversarial stress-testing for proof robustness." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: r,
          disabled: n || !t,
          className: "px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 transition-colors font-medium shadow-sm",
          children: n ? "Attacking..." : "Run Stress Test"
        }
      )
    ] }),
    p && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: p }),
    o && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-900 text-white rounded-lg border border-slate-800 flex flex-col items-center justify-center", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Robustness Score" }),
          /* @__PURE__ */ e.jsxs("span", { className: `text-3xl font-mono mt-1 ${o.is_robust ? "text-emerald-400" : "text-rose-400"}`, children: [
            (o.robustness_score * 100).toFixed(0),
            "%"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-lg border flex flex-col items-center justify-center", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Attacks Blocked" }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-3xl font-mono mt-1 text-slate-700", children: [
            o.attacks_blocked,
            " / ",
            o.total_attacks
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-lg border flex flex-col items-center justify-center", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Status" }),
          /* @__PURE__ */ e.jsx("span", { className: `text-lg font-bold mt-1 uppercase ${o.is_robust ? "text-emerald-600" : "text-rose-600"}`, children: o.is_robust ? "🛡️ Robust" : "⚠️ Vulnerable" })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Red Team Attack Logs" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: o.attack_results.map((c, a) => /* @__PURE__ */ e.jsxs("div", { className: `p-3 border rounded-lg flex items-center gap-4 transition-all ${c.success ? "bg-rose-50 border-rose-200" : "bg-emerald-50 border-emerald-200"}`, children: [
          /* @__PURE__ */ e.jsx("div", { className: `flex-none w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg ${c.success ? "bg-rose-200 text-rose-700" : "bg-emerald-200 text-emerald-700"}`, children: c.success ? "💥" : "🛡️" }),
          /* @__PURE__ */ e.jsxs("div", { className: "flex-1", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-1", children: [
              /* @__PURE__ */ e.jsxs("span", { className: "text-xs font-bold uppercase tracking-tight text-slate-600", children: [
                c.strategy,
                " Attack"
              ] }),
              /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-mono bg-white px-1.5 rounded border", children: [
                "Severity: ",
                c.severity.toFixed(2)
              ] })
            ] }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-700", children: c.description })
          ] })
        ] }, a)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-red-100 rounded-lg bg-red-50/30 text-red-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Adversarial Validation is currently disabled in settings." }) });
}, Ts = ({
  initialTransactions: t = [
    ["Temperature_High", "Mutation_Fast", "Fitness_Improved"],
    ["Temperature_Low", "Mutation_Slow", "Fitness_Stable"],
    ["Temperature_High", "Mutation_Fast", "Diversity_High"],
    ["Temperature_High", "Fitness_Improved", "Zero_Error"]
  ]
}) => {
  const i = v.getState().features.pamiEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(2), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/pami/mine", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            transactions: t,
            min_support: s
          })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Mining failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-end gap-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex-1 space-y-1", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-semibold text-gray-800", children: "Frequent Pattern Discovery" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Uncover hidden associations in evolutionary run data." })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "w-32 space-y-1", children: [
        /* @__PURE__ */ e.jsx("label", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Min Support" }),
        /* @__PURE__ */ e.jsx(
          "input",
          {
            type: "number",
            value: s,
            onChange: (a) => r(parseInt(a.target.value)),
            min: "1",
            className: "w-full p-1.5 border rounded text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
          }
        )
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h,
          className: "px-6 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors font-medium h-[38px]",
          children: h ? "Mining..." : "Mine Patterns"
        }
      )
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center px-1", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest", children: "Mining Results (PAMI)" }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-bold", children: [
          d.total_found,
          " Patterns"
        ] })
      ] }),
      /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 sm:grid-cols-2 gap-3", children: d.patterns.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-slate-50 flex flex-col gap-2 hover:shadow-md transition-shadow", children: [
        /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-1", children: a.items.map((j, k) => /* @__PURE__ */ e.jsx("span", { className: "px-2 py-0.5 bg-white border rounded text-[10px] font-mono text-indigo-600 font-bold", children: j }, k)) }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mt-1 border-t pt-2 border-slate-200", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] text-slate-400 font-bold uppercase", children: "Support Count" }),
          /* @__PURE__ */ e.jsx("span", { className: "text-sm font-bold text-slate-700", children: a.support })
        ] })
      ] }, u)) })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Pattern Mining visualization is currently disabled in settings." }) });
}, Ds = () => {
  const m = v.getState().features.aceEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/ace/analytics");
        if (!s.ok)
          throw new Error("Failed to fetch ACE analytics");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 uppercase tracking-tight", children: "Agentic Context Engine (ACE) Analytics" }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs bg-slate-100 hover:bg-slate-200 px-2 py-1 rounded transition-colors font-bold text-slate-600",
          children: i ? "Refreshing..." : "Refresh Data"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-indigo-600 uppercase tracking-widest border-b pb-1", children: "Top Performing Teams" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-3", children: n.top_teams.map((s, r) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-lg border flex justify-between items-center", children: [
          /* @__PURE__ */ e.jsxs("div", { children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-bold text-slate-700", children: s.team_name }),
            /* @__PURE__ */ e.jsxs("p", { className: "text-[10px] text-slate-400 font-medium", children: [
              "Quality Score: ",
              s.avg_quality_score.toFixed(2)
            ] })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "text-right", children: [
            /* @__PURE__ */ e.jsxs("p", { className: "text-lg font-mono font-bold text-indigo-600", children: [
              (s.success_rate * 100).toFixed(0),
              "%"
            ] }),
            /* @__PURE__ */ e.jsx("p", { className: "text-[8px] uppercase font-bold text-slate-400", children: "Success Rate" })
          ] })
        ] }, r)) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-teal-600 uppercase tracking-widest border-b pb-1", children: "Gauntlet Effectiveness" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-3", children: n.top_gauntlets.map((s, r) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-lg border", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-2", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-bold text-slate-700", children: s.gauntlet_name }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] bg-teal-100 text-teal-700 px-1.5 py-0.5 rounded font-bold uppercase", children: "Active" })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-4", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
              /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-[10px] font-bold text-slate-500", children: [
                /* @__PURE__ */ e.jsx("span", { children: "Detection Rate" }),
                /* @__PURE__ */ e.jsxs("span", { children: [
                  (s.detection_rate * 100).toFixed(0),
                  "%"
                ] })
              ] }),
              /* @__PURE__ */ e.jsx("div", { className: "w-full bg-slate-200 h-1 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-teal-500 h-full", style: { width: `${s.detection_rate * 100}%` } }) })
            ] }),
            /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
              /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-[10px] font-bold text-slate-500", children: [
                /* @__PURE__ */ e.jsx("span", { children: "Precision" }),
                /* @__PURE__ */ e.jsxs("span", { children: [
                  (s.precision * 100).toFixed(0),
                  "%"
                ] })
              ] }),
              /* @__PURE__ */ e.jsx("div", { className: "w-full bg-slate-200 h-1 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-indigo-500 h-full", style: { width: `${s.precision * 100}%` } }) })
            ] })
          ] })
        ] }, r)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "ACE Analytics visualization is currently disabled in settings." }) });
}, Ps = ({
  initialTask: t = ""
}) => {
  const i = v.getState().features.romaEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/roma/solve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task: s })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Recursive solving failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-cyan-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "R" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Recursive Meta-Agents (ROMA)" })
      ] }),
      /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Hierarchical task decomposition and recursive agent orchestration." }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: s,
          onChange: (a) => r(a.target.value),
          placeholder: "Describe a complex multi-step task...",
          className: "w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-cyan-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h || !s,
          className: "px-6 py-2 bg-cyan-700 text-white rounded hover:bg-cyan-800 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
          children: h ? "Orchestrating..." : "Solve Recursively"
        }
      ) })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-cyan-700 uppercase tracking-widest bg-cyan-50 px-2 py-0.5 rounded border border-cyan-100", children: "Synthesized Solution" }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-400 font-mono", children: [
          "Status: ",
          d.status
        ] })
      ] }),
      /* @__PURE__ */ e.jsx("div", { className: "p-4 bg-slate-50 rounded-lg border border-slate-200 shadow-inner", children: /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-700 leading-relaxed whitespace-pre-wrap font-medium", children: d.synthesized_result }) }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-start gap-3 p-3 bg-cyan-50/50 rounded border border-cyan-100/50", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-lg", children: "🤖" }),
        /* @__PURE__ */ e.jsxs("p", { className: "text-[11px] text-cyan-800/80 leading-snug", children: [
          /* @__PURE__ */ e.jsx("strong", { children: "Engine Note:" }),
          " ROMA used recursive planning to break this into atomic subtasks, invoking specialized executors for each, and aggregating them into this final result."
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-cyan-100 rounded-lg bg-cyan-50/30 text-cyan-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "ROMA Orchestration visualization is currently disabled in settings." }) });
}, $s = ({
  initialTask: t = ""
}) => {
  const i = v.getState().features.datapizzaEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/datapizza/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task: s })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Multi-agent execution failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-rose-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "DP" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Multi-Agent Data Processing (DataPizza)" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: s,
          onChange: (a) => r(a.target.value),
          placeholder: "Enter a task for the multi-agent team (e.g., 'Analyze the security of our data transformation logic')...",
          className: "w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-rose-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h || !s,
          className: "px-6 py-2 bg-rose-600 text-white rounded hover:bg-rose-700 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
          children: h ? "Processing..." : "Run Team Workflow"
        }
      ) })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-rose-700 uppercase tracking-widest bg-rose-50 px-2 py-0.5 rounded border border-rose-100", children: [
          "Team: ",
          d.team_name
        ] }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-400 font-mono", children: [
          "Total Steps: ",
          d.total_steps
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [
        d.results.blue && /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-blue-50/50 border-blue-100", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2 mb-2", children: [
            /* @__PURE__ */ e.jsx("span", { className: "w-4 h-4 rounded-full bg-blue-500 block" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-blue-700 uppercase", children: "Blue Team (Solver)" })
          ] }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700 leading-relaxed", children: d.results.blue.response })
        ] }),
        d.results.red && /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-rose-50/50 border-rose-100", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2 mb-2", children: [
            /* @__PURE__ */ e.jsx("span", { className: "w-4 h-4 rounded-full bg-rose-500 block" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-rose-700 uppercase", children: "Red Team (Critiquer)" })
          ] }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700 leading-relaxed", children: d.results.red.response })
        ] }),
        d.results.gold && /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-amber-50/50 border-amber-100", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2 mb-2", children: [
            /* @__PURE__ */ e.jsx("span", { className: "w-4 h-4 rounded-full bg-amber-500 block" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-amber-700 uppercase", children: "Gold Team (Verifier)" })
          ] }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700 leading-relaxed", children: d.results.gold.response })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2 px-1", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-2 h-2 rounded-full bg-emerald-500 animate-pulse" }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-tight", children: [
          "Status: ",
          d.status
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-rose-100 rounded-lg bg-rose-50/30 text-rose-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "DataPizza Multi-Agent visualization is currently disabled in settings." }) });
}, zs = () => {
  const m = v.getState().features.crewaiEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const r = await fetch("/api/openevolve/crewai/summary");
        if (!r.ok)
          throw new Error("Failed to fetch CrewAI summary");
        const c = await r.json();
        d(c);
      } catch (r) {
        b(r instanceof Error ? r.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  if (C(() => {
    m && p();
  }, [m]), !m)
    return /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Project Management visualization is currently disabled in settings." }) });
  const s = (r) => {
    switch (r.toUpperCase()) {
      case "DONE":
        return "bg-emerald-100 text-emerald-700 border-emerald-200";
      case "IN_PROGRESS":
        return "bg-blue-100 text-blue-700 border-blue-200";
      case "TODO":
        return "bg-slate-100 text-slate-700 border-slate-200";
      case "CANCELLED":
        return "bg-rose-100 text-rose-700 border-rose-200";
      default:
        return "bg-slate-50 text-slate-500";
    }
  };
  return /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "H" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "CrewAI Project Tracking" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-blue-600 hover:text-blue-700 transition-colors",
          children: i ? "Syncing..." : "Sync Now"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "lg:col-span-1 space-y-4", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Ticket Overview" }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100 space-y-4", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "text-center pb-4 border-b", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-3xl font-bold text-slate-800", children: n.total_tickets }),
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Active Mappings" })
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-3", children: Object.entries(n.status_distribution).map(([r, c]) => /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-xs font-medium text-slate-600", children: r }),
            /* @__PURE__ */ e.jsx("span", { className: `text-[10px] font-bold px-2 py-0.5 rounded-full ${s(r)}`, children: c })
          ] }, r)) })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "lg:col-span-2 space-y-4", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Recent Activity" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-3", children: n.recent_activity.map((r) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white hover:shadow-md transition-all flex justify-between items-center group", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-3", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-xs font-mono font-bold text-slate-400 group-hover:text-blue-600 transition-colors", children: r.id }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-medium text-slate-700", children: r.task })
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[10px] font-bold px-2 py-1 rounded border ${s(r.status)}`, children: r.status })
        ] }, r.id)) })
      ] })
    ] })
  ] });
}, Os = ({
  initialPrompt: t = "",
  workingDir: m = "."
}) => {
  const h = v.getState().features.claudiomiroEnabled, [n, d] = x(!1), [o, b] = x(null), [p, s] = x(null), [r, c] = x(t), a = async () => {
    if (h) {
      d(!0), s(null);
      try {
        const u = await fetch("/api/openevolve/claudiomiro/decompose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: r, working_dir: m })
        });
        if (!u.ok) {
          const k = await u.json();
          throw new Error(k.detail || "Decomposition failed");
        }
        const j = await u.json();
        b(j);
      } catch (u) {
        s(u instanceof Error ? u.message : "Unknown error");
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-slate-700 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "C" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Autonomous Development (Claudiomiro)" })
      ] }),
      /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Autonomous task decomposition and parallel sub-task generation." }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: r,
          onChange: (u) => c(u.target.value),
          placeholder: "Enter a development task (e.g., 'Refactor the data ingestion pipeline to support Parquet format')...",
          className: "w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-slate-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: a,
          disabled: n || !r,
          className: "px-6 py-2 bg-slate-800 text-white rounded hover:bg-slate-900 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
          children: n ? "Decomposing..." : "Decompose Task"
        }
      ) })
    ] }),
    p && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: p }),
    o && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-slate-700 uppercase tracking-widest bg-slate-100 px-2 py-0.5 rounded border border-slate-200", children: [
          "Task Breakdown: ",
          o.task_id
        ] }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-400 font-mono", children: [
          o.num_tasks,
          " Sub-tasks"
        ] })
      ] }),
      /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: o.sub_tasks.map((u, j) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 border rounded-lg flex flex-col gap-1 hover:bg-white hover:border-slate-300 transition-all group", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800", children: u.title }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[8px] font-bold px-1.5 py-0.5 rounded border uppercase ${u.status === "completed" ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-amber-50 text-amber-600 border-amber-100"}`, children: u.status })
        ] }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500 line-clamp-2", children: u.description })
      ] }, j)) }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-start gap-3 p-3 bg-slate-50/50 rounded border border-slate-100/50", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-lg", children: "🛠️" }),
        /* @__PURE__ */ e.jsxs("p", { className: "text-[11px] text-slate-600 leading-snug italic", children: [
          /* @__PURE__ */ e.jsx("strong", { children: "Claudiomiro" }),
          " has mapped these sub-tasks to a parallel execution DAG. In full autonomous mode, each would be resolved with automated testing and commits."
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Claudiomiro Autonomous Dev visualization is currently disabled in settings." }) });
}, Ms = ({
  task: t,
  output: m
}) => {
  const h = v.getState().features.steerEnabled, [n, d] = x(!1), [o, b] = x(null), [p, s] = x(null), r = async () => {
    if (h) {
      d(!0), s(null);
      try {
        const c = await fetch("/api/openevolve/steer/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task: t, output: m })
        });
        if (!c.ok) {
          const u = await c.json();
          throw new Error(u.detail || "Reliability verification failed");
        }
        const a = await c.json();
        b(a);
      } catch (c) {
        s(c instanceof Error ? c.message : "Unknown error");
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-blue-900 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "S" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Active Reliability (ACE + Steer)" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: r,
          disabled: n || !m,
          className: "px-4 py-2 bg-blue-900 text-white rounded hover:bg-blue-950 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
          children: n ? "Verifying..." : "Verify Output"
        }
      )
    ] }),
    p && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: p }),
    o && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-3 p-3 rounded-lg border bg-slate-50", children: [
        /* @__PURE__ */ e.jsx("div", { className: `w-10 h-10 rounded-full flex items-center justify-center text-xl ${o.all_passed ? "bg-emerald-100 text-emerald-600" : "bg-rose-100 text-rose-600"}`, children: o.all_passed ? "🔒" : "🔓" }),
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsxs("p", { className: "text-sm font-bold text-slate-800", children: [
            "Reality Lock Status: ",
            o.all_passed ? "LOCKED" : "UNLOCKED"
          ] }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: o.all_passed ? "Deterministic quality standards met." : "Verification failed. Closed-loop learning triggered." })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Judge Results" }),
          o.results.map((c, a) => /* @__PURE__ */ e.jsxs("div", { className: "p-2 border rounded bg-white flex justify-between items-center", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-xs font-medium text-slate-700", children: c.judge }),
            /* @__PURE__ */ e.jsx("span", { className: `text-[8px] font-bold px-1.5 py-0.5 rounded border ${c.passed ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-rose-50 text-rose-600 border-rose-100"}`, children: c.passed ? "PASSED" : "FAILED" })
          ] }, a))
        ] }),
        o.ace_learning && /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-indigo-400 uppercase tracking-widest px-1", children: "Closed-Loop Learning" }),
          /* @__PURE__ */ e.jsxs("div", { className: "p-3 border border-indigo-100 bg-indigo-50/30 rounded-lg", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-indigo-700 uppercase mb-1", children: "Skills Acquired:" }),
            /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-1", children: o.ace_learning.learned_skills.map((c) => /* @__PURE__ */ e.jsx("span", { className: "px-2 py-0.5 bg-white border border-indigo-200 rounded text-[10px] text-indigo-600 font-medium", children: c }, c)) })
          ] })
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Active Reliability (Steer) visualization is currently disabled in settings." }) });
}, Ls = () => {
  const m = v.getState().features.researchQuestEnabled, [i, h] = x(!1), [n, d] = x([]), [o, b] = x(null), [p, s] = x(1), r = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const a = await fetch("/api/openevolve/research/stages");
        if (!a.ok)
          throw new Error("Failed to fetch research stages");
        const u = await a.json();
        d(u);
      } catch (a) {
        b(a instanceof Error ? a.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  if (C(() => {
    m && r();
  }, [m]), !m)
    return /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Research Methodology visualization is currently disabled in settings." }) });
  const c = n.find((a) => a.id === p);
  return /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-emerald-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "Q" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Research-Quest methodology" })
      ] }),
      /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-400 uppercase bg-slate-50 px-2 py-1 rounded border", children: "8-Stage Lifecycle" })
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n.length > 0 && /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-4 gap-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsx("div", { className: "lg:col-span-1 flex lg:flex-col gap-2 overflow-x-auto pb-2 lg:pb-0", children: n.map((a) => /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: () => s(a.id),
          className: `flex-none lg:flex-1 text-left px-3 py-2 rounded-lg border transition-all ${p === a.id ? "bg-emerald-600 text-white border-emerald-700 shadow-md transform scale-[1.02]" : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"}`,
          children: /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
            /* @__PURE__ */ e.jsx("span", { className: `w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${p === a.id ? "bg-white/20" : "bg-slate-200"}`, children: a.id }),
            /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold truncate", children: a.name })
          ] })
        },
        a.id
      )) }),
      /* @__PURE__ */ e.jsx("div", { className: "lg:col-span-3 bg-slate-50 rounded-xl border border-slate-200 p-5 space-y-6", children: c && /* @__PURE__ */ e.jsxs(e.Fragment, { children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-xl font-bold text-slate-800", children: c.name }),
          /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-500 mt-1", children: c.description })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-6", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
            /* @__PURE__ */ e.jsx("h5", { className: "text-[10px] font-bold text-emerald-600 uppercase tracking-widest border-b border-emerald-100 pb-1", children: "Stage Objectives" }),
            /* @__PURE__ */ e.jsx("ul", { className: "space-y-2", children: c.objectives.map((a, u) => /* @__PURE__ */ e.jsxs("li", { className: "flex gap-2 items-start text-xs text-slate-700", children: [
              /* @__PURE__ */ e.jsx("span", { className: "text-emerald-500 mt-0.5", children: "●" }),
              a
            ] }, u)) })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
            /* @__PURE__ */ e.jsx("h5", { className: "text-[10px] font-bold text-blue-600 uppercase tracking-widest border-b border-blue-100 pb-1", children: "Expected Outputs" }),
            /* @__PURE__ */ e.jsx("ul", { className: "space-y-2", children: c.outputs.map((a, u) => /* @__PURE__ */ e.jsxs("li", { className: "flex gap-2 items-start text-xs text-slate-700", children: [
              /* @__PURE__ */ e.jsx("span", { className: "text-blue-500 mt-0.5", children: "■" }),
              a
            ] }, u)) })
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-white rounded-lg border border-slate-200 shadow-sm", children: [
          /* @__PURE__ */ e.jsx("h5", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3", children: "Quality Assurance Checks" }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: c.quality_checks.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-3", children: [
            /* @__PURE__ */ e.jsx("div", { className: "w-4 h-4 rounded border border-emerald-200 bg-emerald-50 flex items-center justify-center text-[10px] text-emerald-600", children: "✓" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-xs text-slate-600 font-medium", children: a })
          ] }, u)) })
        ] })
      ] }) })
    ] })
  ] });
}, Fs = ({
  initialText: t = ""
}) => {
  const i = v.getState().features.kgEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/kg/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: s })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "KG generation failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-emerald-700 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "KG" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Knowledge Graph Generation (KG-GEN)" })
      ] }),
      /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Transform unstructured text into a structured knowledge graph." }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: s,
          onChange: (a) => r(a.target.value),
          placeholder: "Enter text to build a graph from...",
          className: "w-full p-3 border rounded-md min-h-[100px] focus:ring-2 focus:ring-emerald-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h || !s,
          className: "px-6 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
          children: h ? "Building Graph..." : "Generate Graph"
        }
      ) })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-emerald-700 uppercase tracking-widest bg-emerald-50 px-2 py-0.5 rounded border border-emerald-100", children: "Graph Entities & Relations" }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-400 font-mono", children: [
          d.nodes.length,
          " Nodes, ",
          d.edges.length,
          " Edges"
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Entities" }),
          /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2", children: d.nodes.map((a, u) => /* @__PURE__ */ e.jsx("span", { className: "px-2 py-1 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded text-xs font-medium", children: a.label }, u)) })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Relations" }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-1", children: d.edges.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-2 bg-slate-50 border rounded text-xs flex justify-between items-center", children: [
            /* @__PURE__ */ e.jsx("span", { className: "font-bold text-slate-700", children: a.source }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-mono text-emerald-600 font-bold px-2", children: a.label }),
            /* @__PURE__ */ e.jsx("span", { className: "font-bold text-slate-700", children: a.target })
          ] }, u)) })
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Knowledge Graph Generation is currently disabled in settings." }) });
}, Gs = () => {
  const m = v.getState().features.sgdEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/sgd/monitoring");
        if (!s.ok)
          throw new Error("Failed to fetch SGD metrics");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    if (m) {
      p();
      const s = setInterval(p, 1e4);
      return () => clearInterval(s);
    }
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white font-bold text-xs shadow-md", children: "S" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Sovereign-Grade Workflow Monitor" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-2 h-2 rounded-full bg-emerald-500 animate-pulse" }),
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-emerald-600 uppercase", children: "Live Metrics" })
      ] })
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 lg:grid-cols-4 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Active Workflows" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-blue-600 mt-1", children: n.active_workflows })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Success Rate" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-2xl font-bold text-emerald-600 mt-1", children: [
            (n.success_rate * 100).toFixed(1),
            "%"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Open Tickets" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-indigo-600 mt-1", children: n.active_tickets })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Gauntlet Runs" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-slate-800 mt-1", children: n.total_gauntlet_runs })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Pipeline Throughput" }),
        /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-slate-50/50", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-xs font-medium mb-2", children: [
              /* @__PURE__ */ e.jsx("span", { className: "text-slate-500", children: "Completed vs Failed Workflows" }),
              /* @__PURE__ */ e.jsxs("span", { className: "text-slate-700 font-bold", children: [
                n.completed_workflows,
                " / ",
                n.failed_workflows
              ] })
            ] }),
            /* @__PURE__ */ e.jsxs("div", { className: "w-full h-2 bg-slate-200 rounded-full overflow-hidden flex", children: [
              /* @__PURE__ */ e.jsx("div", { className: "bg-emerald-500 h-full", style: { width: `${n.completed_workflows / (n.completed_workflows + n.failed_workflows + 0.1) * 100}%` } }),
              /* @__PURE__ */ e.jsx("div", { className: "bg-rose-500 h-full", style: { width: `${n.failed_workflows / (n.completed_workflows + n.failed_workflows + 0.1) * 100}%` } })
            ] })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-slate-50/50", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-xs font-medium mb-2", children: [
              /* @__PURE__ */ e.jsx("span", { className: "text-slate-500", children: "Gauntlet Pass Rate" }),
              /* @__PURE__ */ e.jsxs("span", { className: "text-slate-700 font-bold", children: [
                n.successful_gauntlet_runs,
                " successful"
              ] })
            ] }),
            /* @__PURE__ */ e.jsx("div", { className: "w-full h-2 bg-slate-200 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-indigo-500 h-full", style: { width: `${n.successful_gauntlet_runs / n.total_gauntlet_runs * 100}%` } }) })
          ] })
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "SGD Workflow Monitoring is currently disabled in settings." }) });
}, Vs = ({
  initialContent: t = "",
  iterations: m = 50
}) => {
  const h = v.getState().features.mapElitesEnabled, [n, d] = x(!1), [o, b] = x(null), [p, s] = x(null), [r, c] = x(t), a = async () => {
    if (h) {
      d(!0), s(null);
      try {
        const u = await fetch("/api/openevolve/evolution/map-elites", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: r, iterations: m })
        });
        if (!u.ok) {
          const k = await u.json();
          throw new Error(k.detail || "Evolution failed");
        }
        const j = await u.json();
        b(j);
      } catch (u) {
        s(u instanceof Error ? u.message : "Unknown error");
      } finally {
        d(!1);
      }
    }
  };
  return h ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-gradient-to-tr from-indigo-600 to-violet-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "QD" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Quality-Diversity Optimization (MAP-Elites)" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: r,
          onChange: (u) => c(u.target.value),
          placeholder: "Enter code or content to optimize...",
          className: "w-full p-3 border rounded-md min-h-[80px] focus:ring-2 focus:ring-indigo-500 outline-none text-sm font-sans"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: a,
          disabled: n || !r,
          className: "px-6 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
          children: n ? "Evolving..." : "Run QD Evolution"
        }
      ) })
    ] }),
    p && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: p }),
    o && /* @__PURE__ */ e.jsx("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-6", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsxs("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: [
          "MAP-Elites Archive (",
          o.feature_dimensions.join(" vs "),
          ")"
        ] }),
        /* @__PURE__ */ e.jsx("div", { className: "aspect-square border rounded-lg bg-slate-900 p-2 grid grid-cols-10 grid-rows-10 gap-0.5", children: o.map_elites_grid.flat().map((u, j) => /* @__PURE__ */ e.jsx(
          "div",
          {
            className: "w-full h-full rounded-sm transition-colors hover:ring-1 hover:ring-white cursor-help",
            style: {
              backgroundColor: u > 0 ? `rgba(99, 102, 241, ${u})` : "rgba(30, 41, 59, 0.5)",
              opacity: u > 0 ? 1 : 0.2
            },
            title: `Performance: ${(u * 100).toFixed(1)}%`
          },
          j
        )) }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-center text-slate-400 italic", children: "Heatmap represents high-performing individuals across the feature space." })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Performance over Generations" }),
        /* @__PURE__ */ e.jsxs("div", { className: "h-48 border rounded-lg bg-slate-50 relative flex items-end p-2 gap-1 overflow-hidden", children: [
          o.best_scores.map((u, j) => /* @__PURE__ */ e.jsx(
            "div",
            {
              className: "flex-1 bg-indigo-500 rounded-t-sm min-w-[2px]",
              style: { height: `${u * 100}%` }
            },
            j
          )),
          /* @__PURE__ */ e.jsxs("div", { className: "absolute inset-0 flex flex-col justify-between p-2 pointer-events-none", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-[8px] font-bold text-slate-400 border-b border-dashed w-full", children: "MAX" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[8px] font-bold text-slate-400 border-b border-dashed w-full", children: "AVG" }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[8px] font-bold text-slate-400 w-full", children: "START" })
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-3 gap-2", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "p-2 border rounded bg-white text-center", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Archive" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-mono font-bold text-indigo-600", children: o.map_elites_grid.flat().filter((u) => u > 0).length })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "p-2 border rounded bg-white text-center", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Best" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-mono font-bold text-emerald-600", children: Math.max(...o.best_scores).toFixed(3) })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "p-2 border rounded bg-white text-center", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Diversity" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-mono font-bold text-violet-600", children: o.diversity_scores[o.diversity_scores.length - 1].toFixed(3) })
          ] })
        ] })
      ] })
    ] }) })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-indigo-100 rounded-lg bg-indigo-50/30 text-indigo-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Quality-Diversity visualization is currently disabled in settings." }) });
}, Us = () => {
  const m = v.getState().features.verificationEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/verification/run");
        if (!s.ok) {
          const c = await s.json();
          throw new Error(c.detail || "Verification failed");
        }
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Algorithmic Correctness Analysis" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Formal mathematical verification of core system algorithms." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "px-4 py-2 bg-slate-900 text-white rounded hover:bg-black disabled:opacity-50 transition-colors font-bold text-sm",
          children: i ? "Verifying..." : "Run Analysis"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-4 p-4 bg-slate-900 text-white rounded-xl shadow-lg border border-slate-800", children: [
        /* @__PURE__ */ e.jsxs("div", { className: `w-16 h-16 rounded-full border-4 flex items-center justify-center text-xl font-bold ${n.success_rate === 1 ? "border-emerald-500 text-emerald-400" : "border-rose-500 text-rose-400"}`, children: [
          (n.success_rate * 100).toFixed(0),
          "%"
        ] }),
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest", children: "Verification Summary" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-lg font-medium", children: [
            n.passed,
            " / ",
            n.total_tests,
            " algorithmic properties verified"
          ] }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-xs text-slate-500 mt-1", children: [
            "Last run: ",
            new Date(n.timestamp).toLocaleString()
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Proof Logs" }),
        /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-3", children: n.results.map((s, r) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-slate-50 flex items-center justify-between group hover:bg-white hover:border-slate-300 transition-all", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "space-y-0.5", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase leading-none", children: s.category }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-medium text-slate-700", children: s.test })
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[10px] font-bold px-2 py-0.5 rounded-full border ${s.status === "passed" ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-rose-50 text-rose-600 border-rose-100"}`, children: s.status.toUpperCase() })
        ] }, r)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Mathematical Verification visualization is currently disabled in settings." }) });
}, Is = ({
  initialText: t = ""
}) => {
  const i = v.getState().features.problemAnalysisEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/problem/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ problem_text: s })
        });
        if (!a.ok) {
          const j = await a.json();
          throw new Error(j.detail || "Analysis failed");
        }
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Semantic Problem Analysis" }),
      /* @__PURE__ */ e.jsx(
        "textarea",
        {
          value: s,
          onChange: (a) => r(a.target.value),
          placeholder: "Describe the complex problem to analyze...",
          className: "w-full p-3 border rounded-md min-h-[100px] focus:ring-2 focus:ring-indigo-500 outline-none text-sm"
        }
      ),
      /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: c,
          disabled: h || !s,
          className: "px-6 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 transition-colors font-bold text-sm",
          children: h ? "Analyzing..." : "Perform Analysis"
        }
      ) })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-xl font-bold text-slate-900", children: d.title }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-xs text-slate-500 mt-1 uppercase font-bold tracking-widest", children: [
            d.domain,
            " • ",
            d.problem_type
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "text-right", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Overall Complexity" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-2xl font-mono font-bold text-indigo-600", children: [
            d.complexity.overall.toFixed(1),
            "/10"
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-6", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
          /* @__PURE__ */ e.jsx("h5", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Identified Constraints" }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: d.constraints.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-2 bg-slate-50 border rounded-lg flex flex-col gap-1", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between", children: [
              /* @__PURE__ */ e.jsx("span", { className: "text-[8px] font-bold uppercase text-indigo-500", children: a.type }),
              /* @__PURE__ */ e.jsx("span", { className: `text-[8px] font-bold uppercase ${a.severity === "hard" ? "text-rose-500" : "text-amber-500"}`, children: a.severity })
            ] }),
            /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700", children: a.description })
          ] }, u)) })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
          /* @__PURE__ */ e.jsx("h5", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Success Criteria" }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: d.success_criteria.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-2 border border-emerald-100 bg-emerald-50/30 rounded-lg", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-xs font-bold text-emerald-800", children: a.description }),
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between mt-1", children: [
              /* @__PURE__ */ e.jsx("span", { className: "text-[10px] text-emerald-600/70 font-medium", children: a.metric }),
              /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-emerald-700", children: [
                "Threshold: ",
                a.threshold
              ] })
            ] })
          ] }, u)) })
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Problem Analysis visualization is currently disabled in settings." }) });
}, Ks = ({
  subProblems: t
}) => {
  const i = v.getState().features.dependencyEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), s = async () => {
    if (i) {
      n(!0), p(null);
      try {
        const c = await fetch("/api/openevolve/dependencies/graph", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sub_problems: t })
        });
        if (!c.ok) {
          const u = await c.json();
          throw new Error(u.detail || "Failed to fetch dependency graph");
        }
        const a = await c.json();
        o(a);
      } catch (c) {
        p(c instanceof Error ? c.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  if (C(() => {
    i && t.length > 0 && s();
  }, [i, t]), !i)
    return /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Dependency Mapping visualization is currently disabled in settings." }) });
  const r = (c) => {
    switch (c.toLowerCase()) {
      case "solved":
        return "bg-emerald-500";
      case "in_progress":
        return "bg-blue-500";
      case "pending":
        return "bg-amber-500";
      case "failed":
        return "bg-rose-500";
      default:
        return "bg-slate-400";
    }
  };
  return /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Workflow Dependency Map" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Visualization of sub-problem execution order and complexity." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: s,
          disabled: h,
          className: "text-xs font-bold text-indigo-600 hover:underline",
          children: h ? "Refreshing..." : "Refresh Map"
        }
      )
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsx("div", { className: "p-8 bg-slate-50 rounded-xl border border-slate-100 overflow-x-auto", children: /* @__PURE__ */ e.jsx("div", { className: "flex items-center gap-8 min-w-max", children: d.nodes.map((c, a) => /* @__PURE__ */ e.jsxs(ie.Fragment, { children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col items-center gap-2", children: [
          /* @__PURE__ */ e.jsx("div", { className: `w-16 h-16 rounded-2xl shadow-md flex items-center justify-center text-white font-bold text-xs border-4 border-white ${r(c.status)}`, children: c.id }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-500 uppercase", children: c.status }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-[8px] px-1.5 py-0.5 bg-white border rounded text-slate-400", children: [
            "Comp: ",
            c.complexity
          ] })
        ] }),
        a < d.nodes.length - 1 && /* @__PURE__ */ e.jsx("div", { className: "w-12 h-0.5 bg-slate-200 relative", children: /* @__PURE__ */ e.jsx("div", { className: "absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 rotate-45 border-t-2 border-r-2 border-slate-300" }) })
      ] }, c.id)) }) }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Logic Dependencies" }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-1", children: d.edges.map((c, a) => /* @__PURE__ */ e.jsxs("div", { className: "p-2 border rounded bg-slate-50/50 flex items-center justify-between text-xs", children: [
            /* @__PURE__ */ e.jsx("span", { className: "font-bold text-slate-600", children: c.source }),
            /* @__PURE__ */ e.jsx("span", { className: "text-slate-300", children: "→" }),
            /* @__PURE__ */ e.jsx("span", { className: "font-bold text-slate-800", children: c.target })
          ] }, a)) })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-indigo-50/30 rounded-lg border border-indigo-100", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-indigo-600 uppercase tracking-widest mb-2", children: "Analysis Note" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-[11px] text-indigo-800/80 leading-relaxed italic", children: "This DAG (Directed Acyclic Graph) ensures that prerequisites are satisfied before higher-level integration tasks begin. Nodes are sized by relative complexity and colored by execution status." })
        ] })
      ] })
    ] })
  ] });
}, qs = () => {
  const m = v.getState().features.artifactGraphEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const r = await fetch("/api/openevolve/artifacts/graph");
        if (!r.ok) {
          const a = await r.json();
          throw new Error(a.detail || "Failed to fetch artifact graph");
        }
        const c = await r.json();
        d(c);
      } catch (r) {
        b(r instanceof Error ? r.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  if (C(() => {
    m && p();
  }, [m]), !m)
    return /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Knowledge Artifact Mapping visualization is currently disabled in settings." }) });
  const s = (r) => {
    switch (r.toLowerCase()) {
      case "solution_pattern":
        return "bg-amber-100 text-amber-700 border-amber-200";
      case "team_performance":
        return "bg-blue-100 text-blue-700 border-blue-200";
      case "gauntlet_effectiveness":
        return "bg-rose-100 text-rose-700 border-rose-200";
      default:
        return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };
  return /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Knowledge Artifact Graph" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Relationships between solution patterns, teams, and effectiveness." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-amber-600 hover:underline",
          children: i ? "Analyzing..." : "Refresh Graph"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsx("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "md:col-span-2 p-4 bg-slate-50 rounded-xl border border-slate-100 min-h-[300px] flex flex-col items-center justify-center text-center", children: [
        /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap justify-center gap-4 max-w-md", children: n.nodes.map((r) => /* @__PURE__ */ e.jsxs("div", { className: `px-3 py-2 rounded-lg border shadow-sm ${s(r.type)}`, children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold uppercase opacity-60 tracking-wider", children: r.type.replace("_", " ") }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs font-bold", children: r.label })
        ] }, r.id)) }),
        /* @__PURE__ */ e.jsxs("div", { className: "mt-8", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] text-slate-400 uppercase font-bold tracking-widest", children: "Graph Link Summary" }),
          /* @__PURE__ */ e.jsxs("div", { className: "flex gap-4 mt-2", children: [
            /* @__PURE__ */ e.jsxs("span", { className: "text-xs text-slate-500", children: [
              "Nodes: ",
              /* @__PURE__ */ e.jsx("strong", { children: n.nodes.length })
            ] }),
            /* @__PURE__ */ e.jsxs("span", { className: "text-xs text-slate-500", children: [
              "Relationships: ",
              /* @__PURE__ */ e.jsx("strong", { children: n.edges.length })
            ] })
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Extracted Facts" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2 overflow-y-auto max-h-[300px] pr-2", children: n.edges.map((r, c) => /* @__PURE__ */ e.jsxs("div", { className: "p-2 border rounded bg-white flex flex-col gap-1 shadow-sm", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center text-[10px] text-slate-400 font-bold uppercase", children: [
            /* @__PURE__ */ e.jsx("span", { children: r.source }),
            /* @__PURE__ */ e.jsx("span", { children: r.label })
          ] }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700 font-medium text-center", children: r.target })
        ] }, c)) })
      ] })
    ] }) })
  ] });
}, Ws = () => {
  const m = v.getState().features.sceEnabled, [i, h] = x(!1), [n, d] = x([]), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/constraints/symbolic");
        if (!s.ok) {
          const c = await s.json();
          throw new Error(c.detail || "Failed to fetch symbolic constraints");
        }
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Symbolic Constraint Engine (SCE)" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500 font-medium", children: "Formal logical constraints and proof status." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-indigo-600 hover:underline px-2 py-1",
          children: i ? "Fetching..." : "Sync Constraints"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-3 animate-in fade-in slide-in-from-top-2", children: [
      n.map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all border-l-4 border-l-indigo-500", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start mb-2", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
            /* @__PURE__ */ e.jsx("span", { className: `text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase ${s.type === "hard" ? "bg-rose-50 text-rose-700 border-rose-100" : "bg-blue-50 text-blue-700 border-blue-100"}`, children: s.type }),
            /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800", children: s.id })
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: `flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase border ${s.verified ? "bg-emerald-50 text-emerald-600 border-emerald-200" : "bg-amber-50 text-amber-600 border-amber-200"}`, children: s.verified ? "✓ Verified (Lean4)" : "○ Pending Proof" })
        ] }),
        /* @__PURE__ */ e.jsx("p", { className: "text-sm text-slate-600 mb-3", children: s.description }),
        /* @__PURE__ */ e.jsx("div", { className: "p-2 bg-slate-900 rounded border border-slate-800 overflow-x-auto", children: /* @__PURE__ */ e.jsx("code", { className: "text-[11px] font-mono text-indigo-300", children: s.formalization }) }),
        /* @__PURE__ */ e.jsx("div", { className: "mt-2 flex justify-end", children: /* @__PURE__ */ e.jsxs("span", { className: "text-[9px] text-slate-400 font-medium", children: [
          "Source: ",
          s.source
        ] }) })
      ] }, s.id)),
      !i && n.length === 0 && !o && /* @__PURE__ */ e.jsx("div", { className: "py-12 text-center text-slate-400 border-2 border-dashed rounded-lg", children: "No symbolic constraints found." })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Symbolic Logic visualization is currently disabled in settings." }) });
}, Js = () => {
  const m = v.getState().features.staticAnalysisEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const r = await fetch("/api/openevolve/analysis/static", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_paths: [] })
          // Default to core files
        });
        if (!r.ok) {
          const a = await r.json();
          throw new Error(a.detail || "Analysis failed");
        }
        const c = await r.json();
        d(c);
      } catch (r) {
        b(r instanceof Error ? r.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  if (!m)
    return /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Static Code Analysis is currently disabled in settings." }) });
  const s = (r) => {
    switch (r.toLowerCase()) {
      case "critical":
        return "text-rose-600 bg-rose-50 border-rose-100";
      case "high":
        return "text-orange-600 bg-orange-50 border-orange-100";
      case "medium":
        return "text-amber-600 bg-amber-50 border-amber-100";
      default:
        return "text-slate-600 bg-slate-50 border-slate-100";
    }
  };
  return /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Deep Static Code Analysis" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500 font-medium", children: "Security vulnerability & code quality scanning." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "px-4 py-2 bg-slate-800 text-white rounded hover:bg-black disabled:opacity-50 transition-colors font-bold text-sm",
          children: i ? "Scanning..." : "Run Full Scan"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-900 text-white rounded-xl shadow-md border border-slate-800 text-center", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Total Issues" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-mono font-bold text-emerald-400", children: n.summary.total_issues })
        ] }),
        Object.entries(n.summary.by_severity).map(([r, c]) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-xl border border-slate-100 text-center", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: r }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xl font-bold text-slate-700", children: c })
        ] }, r))
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Top Security & Quality Risks" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: Object.entries(n.issues_by_severity).flatMap(
          ([r, c]) => c.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: `p-3 border rounded-lg flex flex-col gap-1 hover:shadow-sm transition-all border-l-4 ${s(r).split(" ")[2]}`, children: [
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
              /* @__PURE__ */ e.jsx("span", { className: `text-[8px] font-bold px-1.5 py-0.5 rounded border uppercase ${s(r)}`, children: r }),
              /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-400 font-mono", children: [
                a.file,
                ":",
                a.line
              ] })
            ] }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-medium text-slate-800", children: a.message }),
            a.suggestion && /* @__PURE__ */ e.jsxs("p", { className: "text-[11px] text-slate-500 italic mt-1 bg-slate-50/50 p-1 rounded border border-dashed", children: [
              "💡 Suggestion: ",
              a.suggestion
            ] })
          ] }, `${r}-${u}`))
        ) })
      ] })
    ] })
  ] });
}, Qs = () => {
  const m = v.getState().features.lltlEnabled, [i, h] = x(!1), [n, d] = x([]), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/constraints/loss-mapping");
        if (!s.ok)
          throw new Error("Failed to fetch loss mappings");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Logic-to-Loss Translation (LLTL)" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500", children: "Mapping symbolic constraints to differentiable loss functions." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-violet-600 hover:underline",
          children: i ? "Translating..." : "Refresh Mappings"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 animate-in fade-in slide-in-from-top-2", children: n.map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group overflow-hidden", children: [
      /* @__PURE__ */ e.jsx("div", { className: "absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity", children: /* @__PURE__ */ e.jsx("span", { className: "text-4xl font-bold", children: "∫" }) }),
      /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest mb-3", children: s.constraint_id }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-end", children: [
          /* @__PURE__ */ e.jsxs("div", { children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 font-bold uppercase", children: "Relaxation" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-mono font-bold text-violet-600 capitalize", children: s.fuzzy_type })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "text-right", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 font-bold uppercase", children: "Weight" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-lg font-mono font-bold text-slate-800", children: s.weight.toFixed(1) })
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
          /* @__PURE__ */ e.jsx("span", { className: `w-2 h-2 rounded-full ${s.success ? "bg-emerald-500" : "bg-rose-500"}` }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-500 uppercase tracking-wider", children: s.success ? "Differentiable" : "Translation Failed" })
        ] })
      ] }),
      s.error && /* @__PURE__ */ e.jsx("p", { className: "mt-2 text-[10px] text-rose-600 font-medium bg-rose-50 p-1 rounded", children: s.error })
    ] }, s.constraint_id)) }),
    /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-slate-900 rounded-lg border border-slate-800 mt-4", children: /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 font-mono italic", children: "// Foundation: LLTL enables backpropagation through formal logical constraints // by relaxing discrete propositions into smooth barrier and penalty functions." }) })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-violet-100 rounded-lg bg-violet-50/30 text-violet-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Logic-to-Loss visualization is currently disabled in settings." }) });
}, Hs = ({
  workflowId: t
}) => {
  const i = v.getState().features.workflowMonitorEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), s = async () => {
    if (!(!i || !t)) {
      n(!0), p(null);
      try {
        const r = await fetch(`/api/openevolve/workflow/${t}/monitor`);
        if (!r.ok)
          throw new Error("Failed to fetch workflow monitoring data");
        const c = await r.json();
        o(c);
      } catch (r) {
        p(r instanceof Error ? r.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return C(() => {
    if (i && t) {
      s();
      const r = setInterval(s, 5e3);
      return () => clearInterval(r);
    }
  }, [i, t]), i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-indigo-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "W" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Workflow execution monitor" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-slate-400 uppercase", children: [
          "ID: ",
          t
        ] }),
        /* @__PURE__ */ e.jsx("div", { className: "w-2 h-2 rounded-full bg-blue-500 animate-pulse" })
      ] })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Status" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-sm font-bold text-blue-600 mt-1 uppercase", children: d.status })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Progress" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-lg font-bold text-slate-800 mt-1", children: [
            (d.progress * 100).toFixed(1),
            "%"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Runtime" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-lg font-bold text-slate-800 mt-1", children: [
            d.execution_time.toFixed(1),
            "s"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Current Stage" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-sm font-bold text-slate-700 mt-1", children: d.current_stage })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Performance Metrics" }),
          /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-3", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white shadow-sm", children: [
              /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Best Fitness" }),
              /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono font-bold text-emerald-600", children: d.metrics.best_fitness.toFixed(4) })
            ] }),
            /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white shadow-sm", children: [
              /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Avg Fitness" }),
              /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono font-bold text-slate-700", children: d.metrics.avg_fitness.toFixed(4) })
            ] }),
            /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white shadow-sm", children: [
              /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Diversity" }),
              /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono font-bold text-indigo-600", children: d.metrics.diversity.toFixed(4) })
            ] }),
            /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white shadow-sm", children: [
              /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Population" }),
              /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono font-bold text-slate-800", children: d.metrics.population_size })
            ] })
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Resource Utilization" }),
          /* @__PURE__ */ e.jsxs("div", { className: "space-y-3 p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
              /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-[10px] font-bold", children: [
                /* @__PURE__ */ e.jsx("span", { className: "text-slate-500 uppercase", children: "Memory Usage" }),
                /* @__PURE__ */ e.jsxs("span", { className: "text-slate-700", children: [
                  d.resource_usage.memory_mb,
                  " MB"
                ] })
              ] }),
              /* @__PURE__ */ e.jsx("div", { className: "w-full h-1.5 bg-slate-200 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-blue-500 h-full", style: { width: `${Math.min(100, d.resource_usage.memory_mb / 4096 * 100)}%` } }) })
            ] }),
            /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
              /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-[10px] font-bold", children: [
                /* @__PURE__ */ e.jsx("span", { className: "text-slate-500 uppercase", children: "CPU Load" }),
                /* @__PURE__ */ e.jsxs("span", { className: "text-slate-700", children: [
                  (d.resource_usage.cpu_cores * 100).toFixed(0),
                  "%"
                ] })
              ] }),
              /* @__PURE__ */ e.jsx("div", { className: "w-full h-1.5 bg-slate-200 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-indigo-500 h-full", style: { width: `${Math.min(100, d.resource_usage.cpu_cores * 100)}%` } }) })
            ] })
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Live Event Log" }),
        /* @__PURE__ */ e.jsx("div", { className: "bg-slate-900 rounded-lg p-3 space-y-2 max-h-[150px] overflow-y-auto", children: d.events.map((r, c) => /* @__PURE__ */ e.jsxs("div", { className: "flex gap-3 text-[11px] font-mono", children: [
          /* @__PURE__ */ e.jsxs("span", { className: "text-slate-500", children: [
            "[",
            r.timestamp,
            "]"
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: `font-bold ${r.status === "error" ? "text-rose-400" : "text-emerald-400"}`, children: r.status.toUpperCase() }),
          /* @__PURE__ */ e.jsx("span", { className: "text-indigo-200", children: r.message })
        ] }, c)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Workflow Execution Monitor is currently disabled in settings." }) });
}, Bs = () => {
  const m = v.getState().features.lineageEnabled, [i, h] = x(!1), [n, d] = x([]), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/evolution/lineage");
        if (!s.ok)
          throw new Error("Failed to fetch lineage traces");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Evolution Ancestry & Lineage" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-500 font-medium", children: "Ancestral graph of program improvements and generation depth." })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-indigo-600 hover:underline px-2 py-1",
          children: i ? "Tracing..." : "Extract Lineage"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      n.map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-4", children: [
          /* @__PURE__ */ e.jsxs("h4", { className: "text-sm font-bold text-slate-800", children: [
            "Final Candidate: ",
            s.final_program_id
          ] }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full border border-indigo-200 uppercase", children: [
            "Depth: ",
            s.generation_depth,
            " Generations"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "relative", children: [
          /* @__PURE__ */ e.jsx("div", { className: "absolute left-4 top-0 bottom-0 w-0.5 bg-slate-200" }),
          /* @__PURE__ */ e.jsx("div", { className: "space-y-4 relative", children: s.improvement_steps.map((r, c) => /* @__PURE__ */ e.jsxs("div", { className: "flex gap-4 items-start ml-2", children: [
            /* @__PURE__ */ e.jsx("div", { className: "w-4 h-4 rounded-full bg-white border-2 border-indigo-500 flex-none z-10 mt-1" }),
            /* @__PURE__ */ e.jsxs("div", { className: "flex-1 min-w-0", children: [
              /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center text-[10px] font-bold text-slate-400 uppercase mb-1", children: [
                /* @__PURE__ */ e.jsxs("span", { children: [
                  "Step ",
                  r.step,
                  ": ",
                  r.parent_id,
                  " → ",
                  r.child_id
                ] }),
                r.generation && /* @__PURE__ */ e.jsxs("span", { children: [
                  "Gen ",
                  r.generation
                ] })
              ] }),
              /* @__PURE__ */ e.jsx("div", { className: "bg-white border rounded-lg p-2 shadow-sm", children: /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2", children: Object.entries(r.improvement).map(([a, u]) => /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-1.5 px-1.5 py-0.5 bg-emerald-50 border border-emerald-100 rounded text-[9px] font-bold text-emerald-700", children: [
                /* @__PURE__ */ e.jsx("span", { className: "uppercase opacity-60", children: a }),
                /* @__PURE__ */ e.jsxs("span", { children: [
                  "+",
                  u.toFixed(4)
                ] })
              ] }, a)) }) })
            ] })
          ] }, c)) })
        ] })
      ] }, s.final_program_id)),
      !i && n.length === 0 && !o && /* @__PURE__ */ e.jsx("div", { className: "py-12 text-center text-slate-400 border-2 border-dashed rounded-lg", children: "No lineage traces found in checkpoints." })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Evolution Lineage visualization is currently disabled in settings." }) });
}, Ys = () => {
  const m = v.getState().features.gauntletEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/gauntlet/effectiveness", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ gauntlet_ids: ["Red-Team-Core", "Gold-Team-Verify"] })
        });
        if (!s.ok)
          throw new Error("Failed to fetch gauntlet data");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-rose-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "G" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Gauntlet Effectiveness" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-rose-600 hover:underline",
          children: i ? "Analyzing..." : "Refresh Stats"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-2", children: n && Object.values(n).map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative overflow-hidden group", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start mb-4", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800", children: s.gauntlet_id }),
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 uppercase font-bold tracking-widest", children: s.gauntlet_type })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "text-right", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Effectiveness" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-xl font-mono font-bold text-rose-600", children: [
            (s.effectiveness_score * 100).toFixed(1),
            "%"
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-4 mb-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "Catch Rate" }),
          /* @__PURE__ */ e.jsx("div", { className: "w-full bg-slate-200 h-1 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-emerald-500 h-full", style: { width: `${s.avg_catch_rate * 100}%` } }) }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-[10px] font-mono font-bold text-slate-700", children: [
            (s.avg_catch_rate * 100).toFixed(1),
            "%"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "False Positives" }),
          /* @__PURE__ */ e.jsx("div", { className: "w-full bg-slate-200 h-1 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-rose-400 h-full", style: { width: `${s.avg_false_positive_rate * 100}%` } }) }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-[10px] font-mono font-bold text-slate-700", children: [
            (s.avg_false_positive_rate * 100).toFixed(1),
            "%"
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "pt-3 border-t border-slate-200/50 flex justify-between items-center", children: [
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] text-slate-400 font-medium", children: [
          s.total_runs,
          " Total Executions"
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex gap-1", children: [
          /* @__PURE__ */ e.jsx("span", { className: "w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[8px] font-bold text-emerald-600 uppercase", children: "Optimal" })
        ] })
      ] })
    ] }, s.gauntlet_id)) })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-rose-100 rounded-lg bg-rose-50/30 text-rose-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Gauntlet Effectiveness visualization is currently disabled in settings." }) });
}, Xs = () => {
  const m = v.getState().features.patternMiningEnabled, [i, h] = x(!1), [n, d] = x([]), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/patterns/mined");
        if (!s.ok)
          throw new Error("Failed to fetch mined patterns");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-amber-500 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "M" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Solution Pattern Discovery" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-amber-600 hover:underline px-2 py-1",
          children: i ? "Mining..." : "Sync Patterns"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 animate-in fade-in slide-in-from-top-2", children: n.map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all flex flex-col gap-3", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: [
          "Cluster #",
          s.cluster_id
        ] }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full border border-amber-200", children: [
          s.size,
          " Patterns"
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 capitalize", children: s.most_common_domain.replace("_", " ") }),
        /* @__PURE__ */ e.jsxs("p", { className: "text-xs text-slate-500 mt-1 line-clamp-2 italic", children: [
          '"',
          s.description,
          '"'
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-2 pt-2 border-t border-slate-200/50", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Avg Complexity" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-sm font-mono font-bold text-slate-700", children: [
            s.avg_complexity.toFixed(1),
            "/10"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "text-right", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Avg Success" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-sm font-mono font-bold text-emerald-600", children: [
            (s.avg_success_rate * 100).toFixed(0),
            "%"
          ] })
        ] })
      ] })
    ] }, s.cluster_id)) })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-amber-100 rounded-lg bg-amber-50/30 text-amber-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Pattern Mining visualization is currently disabled in settings." }) });
}, Zs = () => {
  const m = v.getState().features.adaptationEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/gauntlet/adaptation");
        if (!s.ok)
          throw new Error("Failed to fetch adaptation statistics");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-gradient-to-br from-rose-500 to-amber-500 flex items-center justify-center text-white font-bold text-xs shadow-md", children: "D" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Dynamic Gauntlet Adaptation" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-2 h-2 rounded-full bg-rose-500 animate-ping" }),
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-rose-600 uppercase", children: "Optimization Engine" })
      ] })
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-900 text-white rounded-xl shadow-lg border border-slate-800 flex flex-col items-center justify-center", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Total Adaptations" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-3xl font-mono font-bold text-rose-400 mt-1", children: n.total_adaptations })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "md:col-span-2 p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 text-center", children: "Strictness Optimization Distribution" }),
          /* @__PURE__ */ e.jsxs("div", { className: "flex items-center h-8 w-full rounded-full overflow-hidden bg-slate-200 shadow-inner", children: [
            /* @__PURE__ */ e.jsx(
              "div",
              {
                className: "bg-rose-500 h-full flex items-center justify-center text-[8px] font-bold text-white transition-all",
                style: { width: `${n.strictness_distribution.more_strict / (n.total_adaptations || 1) * 100}%` },
                title: "More Strict",
                children: n.strictness_distribution.more_strict > 0 && "↑"
              }
            ),
            /* @__PURE__ */ e.jsx(
              "div",
              {
                className: "bg-slate-400 h-full flex items-center justify-center text-[8px] font-bold text-white transition-all",
                style: { width: `${n.strictness_distribution.similar / (n.total_adaptations || 1) * 100}%` },
                title: "Maintained",
                children: n.strictness_distribution.similar > 0 && "•"
              }
            ),
            /* @__PURE__ */ e.jsx(
              "div",
              {
                className: "bg-emerald-500 h-full flex items-center justify-center text-[8px] font-bold text-white transition-all",
                style: { width: `${n.strictness_distribution.less_strict / (n.total_adaptations || 1) * 100}%` },
                title: "Less Strict",
                children: n.strictness_distribution.less_strict > 0 && "↓"
              }
            )
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between mt-2 text-[10px] font-bold text-slate-500", children: [
            /* @__PURE__ */ e.jsxs("span", { children: [
              n.strictness_distribution.more_strict,
              " STRICTOR"
            ] }),
            /* @__PURE__ */ e.jsxs("span", { children: [
              n.strictness_distribution.less_strict,
              " LENIENT"
            ] })
          ] })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Recent Adaptation Events" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: n.recent_events.map((s, r) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white flex items-center justify-between group hover:border-rose-200 transition-colors", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col gap-0.5", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold text-slate-800", children: s.gauntlet }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] text-slate-400 font-mono", children: new Date(s.timestamp).toLocaleTimeString() })
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[9px] font-bold px-2 py-0.5 rounded border uppercase ${s.change === "more_strict" ? "bg-rose-50 text-rose-600 border-rose-100" : s.change === "less_strict" ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-slate-50 text-slate-600 border-slate-100"}`, children: s.change.replace("_", " ") })
        ] }, r)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-rose-100 rounded-lg bg-rose-50/30 text-rose-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Dynamic Adaptation visualization is currently disabled in settings." }) });
}, et = () => {
  const m = v.getState().features.ditoEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/constraints/dito");
        if (!s.ok)
          throw new Error("Failed to run DITO analysis");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-gradient-to-br from-indigo-900 to-slate-900 flex items-center justify-center text-white font-bold text-xs shadow-md", children: "D" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Logic Contradiction Audit (DITO)" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "px-4 py-2 bg-slate-900 text-white rounded hover:bg-black disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
          children: i ? "Analyzing..." : "Run Audit"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100 text-center", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Index Size" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-2xl font-mono font-bold text-slate-800", children: [
            n.total_constraints,
            " Nodes"
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100 text-center", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Query Complexity" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-mono font-bold text-indigo-600", children: "O(log n)" })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: `p-4 rounded-xl border text-center ${n.contradiction_count > 0 ? "bg-rose-50 border-rose-100" : "bg-emerald-50 border-emerald-100"}`, children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Conflicts Found" }),
          /* @__PURE__ */ e.jsx("p", { className: `text-2xl font-mono font-bold ${n.contradiction_count > 0 ? "text-rose-600" : "text-emerald-600"}`, children: n.contradiction_count })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Detected Logical Collisions" }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
          n.contradictions.map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border border-rose-100 bg-rose-50/20 rounded-lg group hover:bg-rose-50 transition-colors", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-1", children: [
              /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-rose-700 uppercase", children: s.pair.join(" ↔ ") }),
              /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-mono font-bold text-rose-400", children: [
                "Confidence: ",
                s.confidence.toFixed(2)
              ] })
            ] }),
            /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700 font-medium leading-relaxed", children: s.description })
          ] }, s.id)),
          n.contradiction_count === 0 && /* @__PURE__ */ e.jsx("div", { className: "py-8 text-center text-slate-400 border-2 border-dashed rounded-lg", children: "No logical contradictions detected by spatial hashing." })
        ] })
      ] }),
      /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-slate-900 rounded-lg border border-slate-800", children: /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center text-[10px] font-mono text-indigo-400/60", children: [
        /* @__PURE__ */ e.jsx("span", { children: "// DITO: Dynamic Inference Trace Optimizer" }),
        /* @__PURE__ */ e.jsx("span", { children: "v1.0.0-stable" })
      ] }) })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-indigo-100 rounded-lg bg-indigo-50/30 text-indigo-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "High-Performance Logic Audit is currently disabled in settings." }) });
}, st = ({
  initialQuery: t = ""
}) => {
  const i = v.getState().features.ragEnabled, [h, n] = x(!1), [d, o] = x([]), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (!(!i || !s)) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/rag/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: s })
        });
        if (!a.ok)
          throw new Error("RAG search failed");
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-sky-700 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "R" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Contextual Knowledge Retrieval (RAG)" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex gap-2", children: [
        /* @__PURE__ */ e.jsx(
          "input",
          {
            type: "text",
            value: s,
            onChange: (a) => r(a.target.value),
            onKeyDown: (a) => a.key === "Enter" && c(),
            placeholder: "Query knowledge base (e.g., 'What are the RESE phase 4 requirements?')...",
            className: "flex-1 p-2 border rounded-md focus:ring-2 focus:ring-sky-500 outline-none text-sm"
          }
        ),
        /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: c,
            disabled: h || !s,
            className: "px-4 py-2 bg-sky-700 text-white rounded hover:bg-sky-800 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
            children: h ? "Searching..." : "Search"
          }
        )
      ] })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d.length > 0 && /* @__PURE__ */ e.jsxs("div", { className: "space-y-3 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Retrieved Context Segments" }),
      d.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all group", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-2", children: [
          /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded-full border border-sky-100", children: [
            "Relevance: ",
            (a.score * 100).toFixed(1),
            "%"
          ] }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-[9px] text-slate-400 font-mono italic", children: [
            "Source: ",
            a.source
          ] })
        ] }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xs text-slate-700 leading-relaxed font-sans", children: a.content })
      ] }, u))
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-sky-100 rounded-lg bg-sky-50/30 text-sky-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Knowledge Retrieval (RAG) visualization is currently disabled in settings." }) });
}, tt = () => {
  const m = v.getState().features.crewaiEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/crewai/monitor");
        if (!s.ok)
          throw new Error("Failed to fetch CrewAI data");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    if (m) {
      p();
      const s = setInterval(p, 8e3);
      return () => clearInterval(s);
    }
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-orange-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "C" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "CrewAI Team Orchestrator" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-orange-600 uppercase tracking-widest", children: (n == null ? void 0 : n.crew_name) || "Autonomous Crew" }),
        /* @__PURE__ */ e.jsx("div", { className: "w-2 h-2 rounded-full bg-orange-500 animate-pulse" })
      ] })
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Active Agents" }),
        /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-3", children: n.agents.map((s, r) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-xl bg-slate-50 flex items-center gap-3 group hover:border-orange-200 transition-colors", children: [
          /* @__PURE__ */ e.jsx("div", { className: "w-10 h-10 rounded-full bg-white border border-slate-200 flex items-center justify-center text-xl grayscale group-hover:grayscale-0 transition-all", children: "🤖" }),
          /* @__PURE__ */ e.jsxs("div", { className: "flex-1 min-w-0", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-bold text-slate-800 truncate", children: s.role }),
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 italic truncate", children: s.goal })
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[8px] font-bold px-1.5 py-0.5 rounded border uppercase ${s.status === "working" ? "bg-orange-50 text-orange-600 border-orange-100 animate-pulse" : "bg-slate-100 text-slate-500 border-slate-200"}`, children: s.status })
        ] }, r)) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center px-1", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest", children: "Orchestration Progress" }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-xs font-bold text-slate-700", children: [
            (n.progress * 100).toFixed(0),
            "%"
          ] })
        ] }),
        /* @__PURE__ */ e.jsx("div", { className: "w-full h-2 bg-slate-100 rounded-full overflow-hidden shadow-inner", children: /* @__PURE__ */ e.jsx("div", { className: "bg-orange-500 h-full transition-all duration-1000", style: { width: `${n.progress * 100}%` } }) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Task Pipeline" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: n.tasks.map((s, r) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white flex items-center justify-between shadow-sm", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col gap-0.5", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-xs font-medium text-slate-700", children: s.description }),
            /* @__PURE__ */ e.jsxs("p", { className: "text-[10px] text-slate-400 font-bold uppercase tracking-tighter", children: [
              "Assigned: ",
              s.agent
            ] })
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[9px] font-bold px-2 py-0.5 rounded border uppercase ${s.status === "done" ? "bg-emerald-50 text-emerald-600 border-emerald-100" : s.status === "in_progress" ? "bg-orange-50 text-orange-600 border-orange-100" : "bg-slate-50 text-slate-400 border-slate-100"}`, children: s.status.replace("_", " ") })
        ] }, r)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-orange-100 rounded-lg bg-orange-50/30 text-orange-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Multi-AI Agent Orchestration is currently disabled in settings." }) });
}, at = ({
  initialText: t = ""
}) => {
  const i = v.getState().features.deepkeEnabled, [h, n] = x(!1), [d, o] = x(null), [b, p] = x(null), [s, r] = x(t), c = async () => {
    if (!(!i || !s)) {
      n(!0), p(null);
      try {
        const a = await fetch("/api/openevolve/knowledge/extract", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: s })
        });
        if (!a.ok)
          throw new Error("DeepKE extraction failed");
        const u = await a.json();
        o(u);
      } catch (a) {
        p(a instanceof Error ? a.message : "Unknown error");
      } finally {
        n(!1);
      }
    }
  };
  return i ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-emerald-700 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "K" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Structured Extraction (DeepKE)" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col gap-2", children: [
        /* @__PURE__ */ e.jsx(
          "textarea",
          {
            value: s,
            onChange: (a) => r(a.target.value),
            placeholder: "Paste text for entity/relation extraction...",
            className: "w-full p-2 border rounded-md focus:ring-2 focus:ring-emerald-500 outline-none text-sm min-h-[80px]"
          }
        ),
        /* @__PURE__ */ e.jsx("div", { className: "flex justify-end", children: /* @__PURE__ */ e.jsx(
          "button",
          {
            onClick: c,
            disabled: h || !s,
            className: "px-4 py-2 bg-emerald-700 text-white rounded hover:bg-emerald-800 disabled:opacity-50 transition-colors font-bold text-sm shadow-sm",
            children: h ? "Extracting..." : "Run DeepKE"
          }
        ) })
      ] })
    ] }),
    b && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: b }),
    d && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2 border-t pt-4", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Discovered Entities" }),
        /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2", children: d.entities.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col items-center p-2 bg-slate-50 border rounded-lg shadow-sm min-w-[80px]", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[8px] font-bold text-emerald-600 uppercase tracking-tighter mb-1", children: a.type }),
          /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold text-slate-800", children: a.text }),
          /* @__PURE__ */ e.jsx("div", { className: "mt-1 w-full bg-slate-200 h-0.5 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "bg-emerald-500 h-full", style: { width: `${a.confidence * 100}%` } }) })
        ] }, u)) })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Semantic Relations" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: d.relations.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-2 border rounded bg-white flex items-center justify-between text-xs shadow-sm", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
            /* @__PURE__ */ e.jsx("span", { className: "font-bold text-slate-700", children: a.head }),
            /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-mono font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100 uppercase", children: a.relation }),
            /* @__PURE__ */ e.jsx("span", { className: "font-bold text-slate-700", children: a.tail })
          ] }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-[9px] text-slate-400 font-mono", children: [
            (a.confidence * 100).toFixed(0),
            "% Match"
          ] })
        ] }, u)) })
      ] }),
      d.events.length > 0 && /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1", children: "Detected Events" }),
        /* @__PURE__ */ e.jsx("div", { className: "space-y-2", children: d.events.map((a, u) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border-l-4 border-l-amber-400 bg-amber-50/20 rounded-r-lg", children: [
          /* @__PURE__ */ e.jsxs("p", { className: "text-xs font-bold text-slate-800 capitalize", children: [
            "Trigger: ",
            a.trigger
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-1 mt-1", children: a.arguments.map((j) => /* @__PURE__ */ e.jsx("span", { className: "text-[9px] bg-white border border-amber-200 text-amber-700 px-1.5 rounded-full font-medium", children: j }, j)) })
        ] }, u)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-emerald-100 rounded-lg bg-emerald-50/30 text-emerald-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Knowledge Extraction (DeepKE) visualization is currently disabled in settings." }) });
}, lt = () => {
  const m = v.getState().features.lean4Enabled, [i, h] = x(!1), [n, d] = x([]), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/mathematics/lean4");
        if (!s.ok)
          throw new Error("Failed to fetch Lean 4 theorems");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "L" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Lean 4 Theorem Prover" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs font-bold text-indigo-600 hover:underline px-2 py-1",
          children: i ? "Refreshing..." : "Sync Theorems"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-4 animate-in fade-in slide-in-from-top-2", children: [
      n.map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 border-l-4 border-l-slate-800", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start mb-3", children: [
          /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-mono font-bold text-slate-900", children: s.name }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[10px] font-bold px-2 py-0.5 rounded-full border ${s.verified ? "bg-emerald-50 text-emerald-600 border-emerald-200" : "bg-amber-50 text-amber-600 border-amber-200"}`, children: s.verified ? "✓ VERIFIED" : "○ UNVERIFIED" })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-white border rounded-lg shadow-inner", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase mb-1", children: "Statement" }),
            /* @__PURE__ */ e.jsx("code", { className: "text-xs text-indigo-700 font-mono leading-relaxed", children: s.statement })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-900 rounded-lg shadow-inner group", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-1", children: [
              /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "Proof Sketch" }),
              /* @__PURE__ */ e.jsx("span", { className: "text-[8px] font-bold text-slate-600 uppercase opacity-0 group-hover:opacity-100 transition-opacity", children: "Tactics Mode" })
            ] }),
            /* @__PURE__ */ e.jsx("pre", { className: "text-xs text-slate-300 font-mono overflow-x-auto", children: s.proof_sketch })
          ] })
        ] })
      ] }, s.name)),
      !i && n.length === 0 && !o && /* @__PURE__ */ e.jsx("div", { className: "py-12 text-center text-slate-400 border-2 border-dashed rounded-lg font-medium italic", children: "No Lean 4 theorems discovered in context." })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-indigo-100 rounded-lg bg-indigo-50/30 text-indigo-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Lean 4 Formal Verification visualization is currently disabled in settings." }) });
}, rt = () => {
  var a, u, j;
  const t = v.getState(), [m, i] = x(null), [h, n] = x(null), [d, o] = x(null), [b, p] = x(null), [s, r] = x(null), c = async () => {
    if (t.features.makerEnabled) {
      const k = await fetch("/api/openevolve/maker/voting");
      i(await k.json());
    }
    if (t.features.mdapEnabled) {
      const k = await fetch("/api/openevolve/mdap/processing");
      n(await k.json());
    }
    if (t.features.mctsEnabled) {
      const k = await fetch("/api/openevolve/mcts/search");
      o(await k.json());
    }
    if (t.features.hybridMCTSEnabled) {
      const k = await fetch("/api/openevolve/mcts/hybrid");
      p(await k.json());
    }
    if (t.features.karateclubEnabled) {
      const k = await fetch("/api/openevolve/graph/ml");
      r(await k.json());
    }
  };
  return C(() => {
    c();
  }, [
    t.features.makerEnabled,
    t.features.mdapEnabled,
    t.features.mctsEnabled,
    t.features.hybridMCTSEnabled,
    t.features.karateclubEnabled
  ]), /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col gap-6 p-4", children: [
    t.features.karateclubEnabled && s && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-white shadow-sm border-slate-200", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-2", children: "KarateClub Graph ML" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold text-indigo-600 uppercase tracking-tighter", children: s.algorithm }),
        /* @__PURE__ */ e.jsxs("span", { className: "text-[10px] font-bold text-slate-400", children: [
          s.clusters,
          " Communities Detected"
        ] })
      ] })
    ] }),
    t.features.hybridMCTSEnabled && b && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-gradient-to-r from-indigo-900 to-slate-900 text-white shadow-lg", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-indigo-300 mb-2", children: "Hybrid MCTS (Synergy)" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "Evolution Count" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-black", children: b.evolution_count })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "text-right", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "Hybrid Score" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-black text-emerald-400", children: (a = b.hybrid_score) == null ? void 0 : a.toFixed(4) })
        ] })
      ] })
    ] }),
    t.features.mdapEnabled && h && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-white shadow-sm", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-2", children: "MDAP Multi-Dim Processing" }),
      /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2", children: (h.dimensions || []).map((k, M) => {
        var $;
        return /* @__PURE__ */ e.jsxs("div", { className: "px-3 py-1 bg-indigo-50 border border-indigo-100 rounded-full flex items-center gap-2", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold text-indigo-700 uppercase", children: k }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-xs font-mono text-indigo-400", children: [
            (((($ = h.scores) == null ? void 0 : $[M]) || 0) * 100).toFixed(0),
            "%"
          ] })
        ] }, k);
      }) })
    ] }),
    t.features.mctsEnabled && d && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-900 text-white shadow-lg", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-indigo-400 mb-2", children: "MCTS Tree Search" }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "Nodes Explored" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-mono font-bold", children: (u = d.nodes_explored) == null ? void 0 : u.toLocaleString() })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "Best Reward" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-mono font-bold text-emerald-400", children: (j = d.best_reward) == null ? void 0 : j.toFixed(4) })
        ] })
      ] })
    ] })
  ] });
}, nt = () => {
  const t = v.getState(), [m, i] = x(null), [h, n] = x(null), [d, o] = x(null), b = async () => {
    if (t.features.e2ePlannerEnabled) {
      const p = await fetch("/api/openevolve/planner/e2e");
      i(await p.json());
    }
    if (t.features.crewaiEnabled) {
      const p = await fetch("/api/openevolve/crewai/summary");
      n(await p.json());
    }
    if (t.features.romaEnabled) {
      const p = await fetch("/api/openevolve/roma/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: "Recursive synthesis of 100+ components" })
      });
      o(await p.json());
    }
  };
  return C(() => {
    b();
  }, [t.features.e2ePlannerEnabled, t.features.crewaiEnabled, t.features.romaEnabled]), /* @__PURE__ */ e.jsxs("div", { className: "p-4 flex flex-col gap-6", children: [
    t.features.romaEnabled && d && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 border-indigo-200", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-2", children: "ROMA Recursive Meta-Agent" }),
      /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 uppercase font-bold mb-3 tracking-widest", children: "Synthesized Response" }),
      /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-white border rounded shadow-inner text-xs text-slate-600 leading-relaxed italic", children: d.synthesized_result })
    ] }),
    t.features.crewaiEnabled && h && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-white shadow-sm", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-4", children: "CrewAI Workflow Status" }),
      /* @__PURE__ */ e.jsx("div", { className: "flex gap-4", children: Object.entries(h.status_distribution || {}).map(([p, s]) => /* @__PURE__ */ e.jsxs("div", { className: "flex-1 p-2 bg-slate-50 rounded border text-center", children: [
        /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: p }),
        /* @__PURE__ */ e.jsx("p", { className: "text-xl font-black text-indigo-600", children: s })
      ] }, p)) })
    ] }),
    t.features.e2ePlannerEnabled && m && /* @__PURE__ */ e.jsxs("div", { className: "p-6 border rounded-2xl bg-white shadow-xl border-indigo-100", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center mb-6", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-xl font-black text-slate-900 tracking-tight", children: "E2E Invention Planner" }),
        /* @__PURE__ */ e.jsx("span", { className: "px-3 py-1 bg-indigo-600 text-white rounded-full text-xs font-bold uppercase tracking-widest animate-pulse", children: "Orchestrating" })
      ] }),
      /* @__PURE__ */ e.jsx("div", { className: "relative h-4 w-full bg-slate-100 rounded-full overflow-hidden mb-8 shadow-inner", children: /* @__PURE__ */ e.jsx(
        "div",
        {
          className: "absolute top-0 left-0 h-full bg-indigo-500 transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(99,102,241,0.5)]",
          style: { width: `${(m.completion || 0) * 100}%` }
        }
      ) }),
      /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: (m.milestones || []).map((p) => /* @__PURE__ */ e.jsxs("div", { className: `p-4 rounded-xl border transition-all ${m.current === p ? "bg-indigo-50 border-indigo-200 shadow-sm scale-105" : "bg-slate-50 border-slate-100 opacity-50"}`, children: [
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase mb-1", children: "Milestone" }),
        /* @__PURE__ */ e.jsx("p", { className: `text-sm font-bold ${m.current === p ? "text-indigo-700" : "text-slate-600"}`, children: p })
      ] }, p)) })
    ] })
  ] });
}, it = () => {
  var r;
  const t = v.getState(), [m, i] = x(null), [h, n] = x(null), [d, o] = x(null), [b, p] = x(null), s = async () => {
    if (t.features.qaSuiteEnabled) {
      const c = await fetch("/api/openevolve/qa/summary");
      i(await c.json());
    }
    if (t.features.redTeamEnabled) {
      const c = await fetch("/api/openevolve/security/red-team");
      n(await c.json());
    }
    if (t.features.blueTeamEnabled) {
      const c = await fetch("/api/openevolve/security/blue-team");
      o(await c.json());
    }
    if (t.features.reseEnabled) {
      const c = await fetch("/api/openevolve/rese/reliability");
      p(await c.json());
    }
  };
  return C(() => {
    s();
  }, [t.features.qaSuiteEnabled, t.features.redTeamEnabled, t.features.blueTeamEnabled, t.features.reseEnabled]), /* @__PURE__ */ e.jsxs("div", { className: "p-4 grid grid-cols-1 md:grid-cols-2 gap-6", children: [
    t.features.qaSuiteEnabled && m && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-white shadow-sm flex flex-col gap-4", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "QA Suite Framework" }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-emerald-50 rounded-lg", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-emerald-600 uppercase", children: "Passed" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-emerald-700", children: m.passed })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-rose-50 rounded-lg", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-rose-600 uppercase", children: "Failed" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-rose-700", children: m.failed })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "text-center text-[10px] font-bold text-slate-400 uppercase", children: [
        "Coverage: ",
        ((m.coverage || 0) * 100).toFixed(1),
        "%"
      ] })
    ] }),
    t.features.reseEnabled && b && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-900 text-white shadow-lg border-indigo-500/30", children: [
      /* @__PURE__ */ e.jsxs("h3", { className: "text-lg font-bold text-indigo-400 mb-4 flex justify-between items-center", children: [
        "RESE Reliability",
        /* @__PURE__ */ e.jsx("span", { className: "w-2 h-2 bg-emerald-500 rounded-full animate-ping" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-end", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-500 uppercase", children: "Score" }),
          /* @__PURE__ */ e.jsx("span", { className: "text-3xl font-mono text-emerald-400 font-bold", children: (r = b.reliability_score) == null ? void 0 : r.toFixed(4) })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-[10px] font-bold text-slate-500 uppercase", children: [
          /* @__PURE__ */ e.jsxs("span", { children: [
            "Error Rate: ",
            b.error_rate
          ] }),
          /* @__PURE__ */ e.jsxs("span", { children: [
            "Uptime: ",
            ((b.uptime || 0) * 100).toFixed(2),
            "%"
          ] })
        ] })
      ] })
    ] }),
    t.features.redTeamEnabled && h && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-rose-950 text-rose-100 border-rose-900 shadow-xl", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold mb-2", children: "Red Team Attacks" }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-end gap-2", children: [
        /* @__PURE__ */ e.jsx("span", { className: "text-4xl font-black", children: h.attacks }),
        /* @__PURE__ */ e.jsx("span", { className: "text-xs font-bold text-rose-400 uppercase pb-1", children: "Attempts Logged" })
      ] }),
      /* @__PURE__ */ e.jsxs("p", { className: "text-[10px] font-bold text-rose-500 mt-2 uppercase tracking-widest", children: [
        "Severity: ",
        h.severity
      ] })
    ] }),
    t.features.blueTeamEnabled && d && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-indigo-950 text-indigo-100 border-indigo-900 shadow-xl", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold mb-2", children: "Blue Team Shields" }),
      /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-2 mt-3", children: (d.defenses || []).map((c) => /* @__PURE__ */ e.jsx("span", { className: "px-2 py-1 bg-indigo-800 rounded text-[10px] font-bold uppercase tracking-tighter", children: c }, c)) })
    ] })
  ] });
}, dt = () => {
  var M, $, A, J;
  const t = v.getState(), [m, i] = x(null), [h, n] = x(null), [d, o] = x(null), [b, p] = x(null), [s, r] = x(null), [c, a] = x(null), [u, j] = x(null), k = async () => {
    if (t.features.materialKGEnabled) {
      const T = await fetch("/api/openevolve/material/kg");
      i(await T.json());
    }
    if (t.features.gnomeEnabled) {
      const T = await fetch("/api/openevolve/discovery/gnome");
      n(await T.json());
    }
    if (t.features.physicsNemoEnabled) {
      const T = await fetch("/api/openevolve/physics/nemo");
      o(await T.json());
    }
    if (t.features.uqEnabled) {
      const T = await fetch("/api/openevolve/uq/analyze");
      p(await T.json());
    }
    if (t.features.pylabrobotEnabled) {
      const T = await fetch("/api/openevolve/robotics/pylabrobot");
      r(await T.json());
    }
    if (t.features.pinnsEnabled) {
      const T = await fetch("/api/openevolve/physics/pinns");
      a(await T.json());
    }
    if (t.features.neuralKGEnabled) {
      const T = await fetch("/api/openevolve/graph/neuralkg");
      j(await T.json());
    }
  };
  return C(() => {
    k();
  }, [
    t.features.materialKGEnabled,
    t.features.gnomeEnabled,
    t.features.physicsNemoEnabled,
    t.features.uqEnabled,
    t.features.pylabrobotEnabled,
    t.features.pinnsEnabled,
    t.features.neuralKGEnabled
  ]), /* @__PURE__ */ e.jsxs("div", { className: "p-4 grid grid-cols-1 gap-6", children: [
    t.features.gnomeEnabled && h && /* @__PURE__ */ e.jsxs("div", { className: "p-6 border rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-700 text-white shadow-xl", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start mb-6", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-xl font-black uppercase tracking-tighter", children: "GNoME Materials Discovery" }),
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold bg-white/20 px-2 py-1 rounded border border-white/30 uppercase", children: "Screening" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 md:grid-cols-3 gap-6", children: [
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-indigo-200 uppercase mb-1", children: "Candidates" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-3xl font-black", children: (M = h.candidate_materials) == null ? void 0 : M.toLocaleString() })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-indigo-200 uppercase mb-1", children: "Validated" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-3xl font-black text-emerald-300", children: h.valid_materials })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "hidden md:block", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-indigo-200 uppercase mb-1", children: "Success Rate" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-3xl font-black text-white/80", children: [
            (h.valid_materials / (h.candidate_materials || 1) * 100).toFixed(2),
            "%"
          ] })
        ] })
      ] })
    ] }),
    /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6", children: [
      t.features.pylabrobotEnabled && s && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 border-slate-200 shadow-sm", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-2", children: "PyLabRobot Automation" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
          /* @__PURE__ */ e.jsxs("div", { children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Status" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-sm font-bold text-emerald-600 uppercase", children: s.status })
          ] }),
          /* @__PURE__ */ e.jsxs("div", { className: "text-right", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Plates" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-xl font-black text-slate-800", children: s.plates })
          ] })
        ] })
      ] }),
      t.features.pinnsEnabled && c && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-white shadow-sm border-indigo-100", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-2", children: "PINNs Physics ML" }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "PDE Residual" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-lg font-mono font-bold text-indigo-600", children: c.pde_residual })
        ] })
      ] }),
      t.features.neuralKGEnabled && u && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-900 text-white shadow-lg", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-indigo-400 mb-2", children: "NeuralKG Embedding" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-500 uppercase", children: u.algorithm }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-xl font-black text-emerald-400", children: [
            u.dim,
            "d"
          ] })
        ] })
      ] })
    ] }),
    t.features.uqEnabled && b && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-white shadow-sm border-indigo-100", children: [
      /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-4", children: "Uncertainty Quantification (UQ)" }),
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-2 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-lg", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Variance" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono font-bold text-indigo-600", children: (A = ($ = b.statistics) == null ? void 0 : $.std) == null ? void 0 : A.toFixed(6) })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-3 bg-slate-50 rounded-lg", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Confidence" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-xl font-mono font-bold text-emerald-600", children: "0.99" })
        ] })
      ] })
    ] }),
    /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-6", children: [
      t.features.materialKGEnabled && m && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-white shadow-sm border-slate-200", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-4", children: "Material Knowledge Graph" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "text-center", children: [
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase", children: "Compounds" }),
            /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-slate-800", children: (J = m.compounds) == null ? void 0 : J.toLocaleString() })
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: "flex flex-wrap gap-1 justify-end max-w-[150px]", children: (m.properties || []).map((T) => /* @__PURE__ */ e.jsx("span", { className: "px-2 py-0.5 bg-slate-100 rounded text-[8px] font-bold text-slate-500 uppercase", children: T }, T)) })
        ] })
      ] }),
      t.features.physicsNemoEnabled && d && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 border-slate-200 shadow-sm", children: [
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 mb-4", children: "Physics-NeMo Simulation" }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-3", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center text-xs", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-slate-500 font-bold uppercase", children: "Convergence" }),
            /* @__PURE__ */ e.jsxs("span", { className: "font-mono font-bold text-indigo-600", children: [
              ((d.convergence || 0) * 100).toFixed(2),
              "%"
            ] })
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: "w-full h-1 bg-slate-200 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx("div", { className: "h-full bg-indigo-500", style: { width: `${(d.convergence || 0) * 100}%` } }) }),
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center text-[10px]", children: [
            /* @__PURE__ */ e.jsxs("span", { className: "text-slate-400 font-medium", children: [
              "Error Norm: ",
              d.error_norm
            ] }),
            /* @__PURE__ */ e.jsxs("span", { className: "text-slate-400 font-medium", children: [
              "Sims: ",
              d.simulations
            ] })
          ] })
        ] })
      ] })
    ] })
  ] });
}, ot = () => {
  const t = v.getState();
  return t.features.autogptEnabled || t.features.autogenEnabled || t.features.metagptEnabled || t.features.aiScientistEnabled || t.features.uncertainpyEnabled || t.features.riskAnalyzerEnabled || t.features.llm4iasEnabled || t.features.claraverseEnabled ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-gradient-to-tr from-purple-600 to-pink-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "9" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Roadmap Agent Control Plane" })
      ] }),
      /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-slate-400 uppercase bg-slate-50 px-2 py-1 rounded border", children: "Category 9 Synergy" })
    ] }),
    /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [
      t.features.autogptEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "AutoGPT Swarm" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "Autonomous Task Loops" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
          /* @__PURE__ */ e.jsx("span", { className: "w-2 h-2 bg-emerald-500 rounded-full animate-pulse" }),
          /* @__PURE__ */ e.jsxs("span", { className: "text-xs font-mono text-slate-600 tracking-tighter", children: [
            "SWARM_ACTIVE // LOOP_ID: ",
            Math.random().toString(36).substring(7)
          ] })
        ] })
      ] }),
      t.features.autogenEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "Microsoft AutoGen" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "Conversation Dynamics" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex gap-1", children: [
          /* @__PURE__ */ e.jsx("div", { className: "w-6 h-6 rounded bg-blue-100 flex items-center justify-center text-[10px]", children: "A1" }),
          /* @__PURE__ */ e.jsx("div", { className: "w-6 h-6 rounded bg-indigo-100 flex items-center justify-center text-[10px]", children: "A2" }),
          /* @__PURE__ */ e.jsx("div", { className: "w-6 h-6 rounded bg-violet-100 flex items-center justify-center text-[10px]", children: "A3" })
        ] })
      ] }),
      t.features.metagptEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "MetaGPT Firm" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "Software Company Simulation" }),
        /* @__PURE__ */ e.jsxs("div", { className: "space-y-1", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between text-[8px] font-bold text-slate-400", children: [
            /* @__PURE__ */ e.jsx("span", { children: "PROJECT_ALPHA" }),
            /* @__PURE__ */ e.jsx("span", { children: "85%" })
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: "w-full h-1 bg-slate-200 rounded-full", children: /* @__PURE__ */ e.jsx("div", { className: "h-full bg-indigo-500 w-[85%]" }) })
        ] })
      ] }),
      t.features.aiScientistEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "AI Scientist" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "Automated Hypothesizing" }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-2 bg-white rounded border border-dashed flex flex-col gap-1", children: [
          /* @__PURE__ */ e.jsx("span", { className: "text-[9px] font-bold text-indigo-600 uppercase", children: "New Hypothesis" }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] text-slate-600 italic leading-tight", children: "Neural-topological alignment improves zero-shot transfer." })
        ] })
      ] }),
      t.features.uncertainpyEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "Uncertainty Analysis" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "Sensitivity Propagation" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex items-center justify-between", children: [
          /* @__PURE__ */ e.jsx("div", { className: "flex gap-0.5 items-end h-6", children: [0.4, 0.7, 0.3, 0.9, 0.5].map((i, h) => /* @__PURE__ */ e.jsx("div", { className: "w-2 bg-indigo-400 rounded-t-sm", style: { height: `${i * 100}%` } }, h)) }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-mono text-slate-600", children: "Var: 0.02" })
        ] })
      ] }),
      t.features.riskAnalyzerEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "LLM Risk Analyzer" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "Vulnerability Detection" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
          /* @__PURE__ */ e.jsx("div", { className: "w-3 h-3 rounded-full bg-emerald-500 shadow-sm" }),
          /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-emerald-600 uppercase", children: "Status: Secure" })
        ] })
      ] }),
      t.features.llm4iasEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "SOP Optimization" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "Procedure Enhancement" }),
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-indigo-600 tracking-tighter uppercase", children: "+15.2% Efficiency Gain" })
      ] }),
      t.features.claraverseEnabled && /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 relative group", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800 mb-1", children: "Integration Assessment" }),
        /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-500 mb-3 uppercase font-bold", children: "ClaraVerse Compatibility" }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center text-[10px] font-bold text-slate-400", children: [
          /* @__PURE__ */ e.jsx("span", { children: "92% Verified" }),
          /* @__PURE__ */ e.jsx("span", { className: "w-1.5 h-1.5 rounded-full bg-emerald-400" })
        ] })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Roadmap Agent (Category 9) visualizations are currently disabled in settings." }) });
}, ct = () => {
  const m = v.getState().features.collaborationEnabled, [i, h] = x(!1), [n, d] = x([]), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/collaboration/sessions");
        if (!s.ok)
          throw new Error("Failed to fetch collaboration sessions");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    if (m) {
      p();
      const s = setInterval(p, 5e3);
      return () => clearInterval(s);
    }
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-4 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-sky-600 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "H" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800 tracking-tight", children: "Collaboration Hub" })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-2 h-2 rounded-full bg-emerald-500 animate-pulse" }),
        /* @__PURE__ */ e.jsx("span", { className: "text-[10px] font-bold text-emerald-600 uppercase", children: "Live Sync" })
      ] })
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    /* @__PURE__ */ e.jsxs("div", { className: "space-y-3 animate-in fade-in slide-in-from-top-2", children: [
      n.map((s) => /* @__PURE__ */ e.jsxs("div", { className: "p-4 border rounded-xl bg-slate-50 hover:bg-white hover:shadow-md transition-all", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-start mb-3", children: [
          /* @__PURE__ */ e.jsxs("div", { children: [
            /* @__PURE__ */ e.jsx("h4", { className: "text-sm font-bold text-slate-800", children: s.name }),
            /* @__PURE__ */ e.jsx("p", { className: "text-[10px] text-slate-400 font-mono", children: s.session_id })
          ] }),
          /* @__PURE__ */ e.jsx("span", { className: `text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${s.status === "active" ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-amber-50 text-amber-600 border-amber-100"}`, children: s.status })
        ] }),
        /* @__PURE__ */ e.jsx("div", { className: "flex items-center gap-2 mb-4 overflow-x-auto pb-1", children: s.participants.map((r) => /* @__PURE__ */ e.jsxs("div", { className: "flex-none px-2 py-1 bg-white border rounded text-[10px] font-bold text-slate-600 flex items-center gap-1.5 shadow-sm", children: [
          /* @__PURE__ */ e.jsx("span", { className: "w-1.5 h-1.5 rounded-full bg-sky-400" }),
          r
        ] }, r)) }),
        /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center pt-3 border-t border-slate-200/50", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex gap-4", children: [
            /* @__PURE__ */ e.jsxs("div", { className: "text-center", children: [
              /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Conflicts" }),
              /* @__PURE__ */ e.jsx("p", { className: `text-xs font-bold ${s.conflict_count > 0 ? "text-rose-500" : "text-slate-600"}`, children: s.conflict_count })
            ] }),
            /* @__PURE__ */ e.jsxs("div", { className: "text-center", children: [
              /* @__PURE__ */ e.jsx("p", { className: "text-[8px] font-bold text-slate-400 uppercase", children: "Last Activity" }),
              /* @__PURE__ */ e.jsx("p", { className: "text-xs font-bold text-slate-600", children: s.last_edit })
            ] })
          ] }),
          /* @__PURE__ */ e.jsx("button", { className: "px-3 py-1 bg-sky-600 text-white text-[10px] font-bold rounded hover:bg-sky-700 transition-colors shadow-sm", children: "JOIN SESSION" })
        ] })
      ] }, s.session_id)),
      !i && n.length === 0 && !o && /* @__PURE__ */ e.jsx("div", { className: "py-12 text-center text-slate-400 border-2 border-dashed rounded-lg", children: "No active collaboration sessions found." })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-sky-100 rounded-lg bg-sky-50/30 text-sky-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Multi-Agent Collaboration visualization is currently disabled in settings." }) });
}, xt = () => {
  const m = v.getState().features.globalAnalyticsEnabled, [i, h] = x(!1), [n, d] = x(null), [o, b] = x(null), p = async () => {
    if (m) {
      h(!0), b(null);
      try {
        const s = await fetch("/api/openevolve/analytics/global");
        if (!s.ok)
          throw new Error("Failed to fetch global analytics");
        const r = await s.json();
        d(r);
      } catch (s) {
        b(s instanceof Error ? s.message : "Unknown error");
      } finally {
        h(!1);
      }
    }
  };
  return C(() => {
    m && p();
  }, [m]), m ? /* @__PURE__ */ e.jsxs("div", { className: "flex flex-col space-y-6 p-4 border rounded-lg bg-white shadow-sm", children: [
    /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "flex items-center gap-2", children: [
        /* @__PURE__ */ e.jsx("div", { className: "w-8 h-8 rounded bg-slate-900 flex items-center justify-center text-white font-bold text-xs shadow-sm", children: "A" }),
        /* @__PURE__ */ e.jsx("h3", { className: "text-lg font-bold text-slate-800", children: "Global System Performance" })
      ] }),
      /* @__PURE__ */ e.jsx(
        "button",
        {
          onClick: p,
          disabled: i,
          className: "text-xs bg-slate-100 hover:bg-slate-200 px-2 py-1 rounded transition-colors font-bold text-slate-600",
          children: i ? "Aggregating..." : "Refresh Summary"
        }
      )
    ] }),
    o && /* @__PURE__ */ e.jsx("div", { className: "p-3 bg-red-50 text-red-700 border border-red-200 rounded text-sm", children: o }),
    n && /* @__PURE__ */ e.jsxs("div", { className: "space-y-6 animate-in fade-in slide-in-from-top-2", children: [
      /* @__PURE__ */ e.jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-900 text-white rounded-xl shadow-lg", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Total Cumulative Cost" }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-3xl font-mono font-bold text-emerald-400 mt-1", children: [
            "$",
            n.total_cost.toFixed(2)
          ] })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Global Token Usage" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-slate-800 mt-1", children: n.total_tokens.toLocaleString() })
        ] }),
        /* @__PURE__ */ e.jsxs("div", { className: "p-4 bg-slate-50 rounded-xl border border-slate-100", children: [
          /* @__PURE__ */ e.jsx("p", { className: "text-[10px] font-bold text-slate-400 uppercase tracking-widest", children: "Workflows Tracked" }),
          /* @__PURE__ */ e.jsx("p", { className: "text-2xl font-bold text-slate-800 mt-1", children: n.total_workflows })
        ] })
      ] }),
      /* @__PURE__ */ e.jsxs("div", { className: "space-y-4", children: [
        /* @__PURE__ */ e.jsx("h4", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest px-1", children: "Provider Cost Breakdown" }),
        /* @__PURE__ */ e.jsx("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4", children: Object.entries(n.provider_breakdown).map(([s, r]) => /* @__PURE__ */ e.jsxs("div", { className: "p-3 border rounded-lg bg-white shadow-sm flex flex-col gap-2", children: [
          /* @__PURE__ */ e.jsxs("div", { className: "flex justify-between items-center", children: [
            /* @__PURE__ */ e.jsx("span", { className: "text-sm font-bold text-slate-700 capitalize", children: s }),
            /* @__PURE__ */ e.jsxs("span", { className: "text-xs font-mono font-bold text-emerald-600", children: [
              "$",
              r.cost.toFixed(4)
            ] })
          ] }),
          /* @__PURE__ */ e.jsx("div", { className: "w-full bg-slate-100 h-1.5 rounded-full overflow-hidden", children: /* @__PURE__ */ e.jsx(
            "div",
            {
              className: "bg-indigo-500 h-full",
              style: { width: `${Math.min(100, r.cost / n.total_cost * 100)}%` }
            }
          ) }),
          /* @__PURE__ */ e.jsxs("p", { className: "text-[10px] text-slate-400 font-medium", children: [
            r.tokens.toLocaleString(),
            " tokens utilized"
          ] })
        ] }, s)) })
      ] })
    ] })
  ] }) : /* @__PURE__ */ e.jsx("div", { className: "p-6 text-center border-2 border-dashed border-slate-100 rounded-lg bg-slate-50/30 text-slate-400", children: /* @__PURE__ */ e.jsx("p", { className: "font-medium italic", children: "Global Performance Analytics is currently disabled in settings." }) });
};
export {
  Ds as ACEViz,
  Zs as AdaptationViz,
  As as AdversarialViz,
  qs as ArtifactViz,
  vs as CausalDiscoveryViz,
  Es as ChemicalViz,
  Os as ClaudiomiroViz,
  ct as CollabViz,
  tt as CrewAIViz,
  et as DITOViz,
  $s as DataPizzaViz,
  at as DeepKEViz,
  Ks as DependencyViz,
  ks as ExperimentViz,
  Cs as ExtractionViz,
  Ys as GauntletViz,
  xt as GlobalAnalyticsViz,
  zs as CrewAIViz,
  Fs as KGViz,
  Qs as LLTLViz,
  lt as Lean4Viz,
  _s as LeanAideViz,
  Bs as LineageViz,
  Vs as MAPElitesViz,
  rt as MakerViz,
  Ns as OptimizationViz,
  nt as OrchestratorViz,
  Ts as PAMIViz,
  Xs as PatternMinerViz,
  Is as ProblemAnalysisViz,
  js as PyGraphistryViz,
  it as QAComplianceViz,
  st as RAGViz,
  Ps as ROMAViz,
  Ls as ResearchQuestViz,
  ot as RoadmapAgentViz,
  Ws as SCEViz,
  Gs as SGDViz,
  Rs as SOPViz,
  dt as ScientificDiscoveryViz,
  Js as StaticAnalysisViz,
  Ms as SteerViz,
  Ss as TemporalGraphViz,
  ws as UQViz,
  Us as VerificationViz,
  ys as VizSettingsPanel,
  Hs as WorkflowMonitorViz,
  hs as createPyGraphistryPlugin,
  v as pygraphistryPlugin
};
