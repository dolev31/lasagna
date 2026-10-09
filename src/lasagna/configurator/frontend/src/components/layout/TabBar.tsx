import React from "react";
import type { TabDef } from "../../api/types";

interface Props {
  tabs: TabDef[];
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export default function TabBar({ tabs, activeTab, onTabChange }: Props) {
  return (
    <div className="cfg-tabbar">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          className={`cfg-tab ${activeTab === tab.key ? "active" : ""}`}
          onClick={() => onTabChange(tab.key)}
          title={tab.description}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
