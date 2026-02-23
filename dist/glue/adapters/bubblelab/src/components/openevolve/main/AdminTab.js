"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdminTab = void 0;
const react_1 = __importDefault(require("react"));
const tabs_1 = require("@/components/ui/tabs");
const TeamManagerTab_1 = require("./TeamManagerTab");
const GauntletDesignerTab_1 = require("./GauntletDesignerTab");
const AdminTab = () => {
    return (<div className="space-y-6">
      <tabs_1.Tabs defaultValue="teams" className="w-full">
        <tabs_1.TabsList className="grid w-full grid-cols-2">
          <tabs_1.TabsTrigger value="teams">Teams</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="gauntlets">Gauntlets</tabs_1.TabsTrigger>
        </tabs_1.TabsList>
        <tabs_1.TabsContent value="teams" className="pt-6">
          <TeamManagerTab_1.TeamManagerTab />
        </tabs_1.TabsContent>
        <tabs_1.TabsContent value="gauntlets" className="pt-6">
          <GauntletDesignerTab_1.GauntletDesignerTab />
        </tabs_1.TabsContent>
      </tabs_1.Tabs>
    </div>);
};
exports.AdminTab = AdminTab;
//# sourceMappingURL=AdminTab.js.map