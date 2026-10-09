import React, { useState, useMemo } from "react";
import { useConfig } from "./hooks/useConfig";
import * as api from "./api/client";

import Header from "./components/layout/Header";
import TabBar from "./components/layout/TabBar";
import Sidebar from "./components/layout/Sidebar";
import Footer from "./components/layout/Footer";
import ReindexBanner from "./components/shared/ReindexBanner";
import Toast from "./components/shared/Toast";
import CreateProfileDialog from "./components/shared/CreateProfileDialog";
import ConfirmDialog from "./components/shared/ConfirmDialog";

import EssentialTab from "./components/tabs/EssentialTab";
import SearchTab from "./components/tabs/SearchTab";
import ProcessingTab from "./components/tabs/ProcessingTab";
import AdvancedTab from "./components/tabs/AdvancedTab";

import type { TabDef, ParameterDef } from "./api/types";

const DEFAULT_TABS: TabDef[] = [
  { key: "essential", label: "Essential", description: "Core settings that cover 80% of use cases" },
  { key: "search", label: "Search & Retrieval", description: "Fine-tune search behavior and ranking" },
  { key: "processing", label: "Processing", description: "Document chunking, OCR, and awareness" },
  { key: "advanced", label: "Advanced", description: "Agentic RAG, citations, caching, observability" },
];

export default function App() {
  const state = useConfig();
  const [activeTab, setActiveTab] = useState("essential");
  const [showSaveAs, setShowSaveAs] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const tabs = state.schema?.tabs ?? DEFAULT_TABS;
  const params = state.schema?.parameters ?? [];

  // Filter params by active tab, excluding hidden params
  const tabParams = useMemo(
    () => params.filter((p: ParameterDef) => p.tab === activeTab && !p.hidden),
    [params, activeTab],
  );

  if (state.loading && !state.profiles.length) {
    return (
      <div className="cfg-app">
        <div className="cfg-loading">
          <div className="cfg-spinner" />
          <span>Loading configuration...</span>
        </div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="cfg-app">
        <div className="cfg-error-state">
          <div className="cfg-error-icon">!</div>
          <div className="cfg-error-title">Unable to load configuration</div>
          <div className="cfg-error-message">{state.error}</div>
          <button className="cfg-btn primary" onClick={state.reload}>Try Again</button>
        </div>
      </div>
    );
  }

  const handleRebuildIndex = async () => {
    try {
      await api.rebuildIndex();
    } catch {
      // handled by toast
    }
  };

  return (
    <div className="cfg-app">
      <Toast />

      <Header
        onExport={state.exportConfig}
        onImport={state.importConfig}
      />

      {state.reindexNeeded && (
        <ReindexBanner
          changedKeys={state.reindexChangedKeys}
          onRebuild={handleRebuildIndex}
        />
      )}

      <div className="cfg-body">
        <Sidebar
          profiles={state.profiles}
          activeProfile={state.activeProfile}
          onSelectProfile={state.setActiveProfile}
          onNewProfile={() => setShowSaveAs(true)}
          onDeleteProfile={(name) => setDeleteConfirm(name)}
          config={state.config}
          originalConfig={state.originalConfig}
          schemaParams={params}
        />

        <div className="cfg-main">
          <TabBar
            tabs={tabs}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />

          <div className="cfg-content">
            {activeTab === "essential" && (
              <EssentialTab config={state.config} onUpdate={state.updateParam} params={tabParams} />
            )}
            {activeTab === "search" && (
              <SearchTab config={state.config} onUpdate={state.updateParam} params={tabParams} />
            )}
            {activeTab === "processing" && (
              <ProcessingTab config={state.config} onUpdate={state.updateParam} params={tabParams} />
            )}
            {activeTab === "advanced" && (
              <AdvancedTab config={state.config} onUpdate={state.updateParam} params={tabParams} />
            )}
          </div>
        </div>
      </div>

      <Footer
        unsavedCount={state.unsavedCount}
        reindexNeeded={state.reindexNeeded}
        saveStatus={state.saveStatus}
        isProtected={state.isProtected}
        onSave={state.save}
        onReset={state.reset}
        onSaveAs={() => setShowSaveAs(true)}
        onRebuildIndex={handleRebuildIndex}
      />

      {showSaveAs && (
        <CreateProfileDialog
          onSave={(name) => {
            state.saveAs(name);
            setShowSaveAs(false);
          }}
          onCancel={() => setShowSaveAs(false)}
        />
      )}

      {deleteConfirm && (
        <ConfirmDialog
          title="Delete Profile"
          message={`Are you sure you want to delete "${deleteConfirm}"? This cannot be undone.`}
          confirmLabel="Delete"
          danger
          onConfirm={() => {
            state.deleteProfile(deleteConfirm);
            setDeleteConfirm(null);
          }}
          onCancel={() => setDeleteConfirm(null)}
        />
      )}
    </div>
  );
}
