import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TeamManagerTab } from "./TeamManagerTab";
import { GauntletDesignerTab } from "./GauntletDesignerTab";

export const AdminTab: React.FC = () => {
  return (
    <div className="space-y-6">
      <Tabs defaultValue="teams" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="teams">Teams</TabsTrigger>
          <TabsTrigger value="gauntlets">Gauntlets</TabsTrigger>
        </TabsList>
        <TabsContent value="teams" className="pt-6">
          <TeamManagerTab />
        </TabsContent>
        <TabsContent value="gauntlets" className="pt-6">
          <GauntletDesignerTab />
        </TabsContent>
      </Tabs>
    </div>
  );
};
