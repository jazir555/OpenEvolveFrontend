"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createRomaPluginFactory = exports.romaPlugin = exports.createRomaPlugin = exports.RomaConfigPanel = void 0;
__exportStar(require("./types/plugin-types"), exports);
var RomaConfigPanel_1 = require("./components/RomaConfigPanel");
Object.defineProperty(exports, "RomaConfigPanel", { enumerable: true, get: function () { return __importDefault(RomaConfigPanel_1).default; } });
var createRomaPlugin_1 = require("./utils/createRomaPlugin");
Object.defineProperty(exports, "createRomaPlugin", { enumerable: true, get: function () { return createRomaPlugin_1.createRomaPlugin; } });
Object.defineProperty(exports, "romaPlugin", { enumerable: true, get: function () { return createRomaPlugin_1.romaPlugin; } });
var createRomaPlugin_2 = require("./utils/createRomaPlugin");
Object.defineProperty(exports, "createRomaPluginFactory", { enumerable: true, get: function () { return __importDefault(createRomaPlugin_2).default; } });
//# sourceMappingURL=index.js.map