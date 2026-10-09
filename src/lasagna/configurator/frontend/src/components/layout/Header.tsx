import React, { useRef } from "react";
import { BookOpen, Download, Upload } from "lucide-react";

interface Props {
  onExport: () => void;
  onImport: (data: unknown) => void;
}

export default function Header({ onExport, onImport }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

  const handleImportClick = () => fileRef.current?.click();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string);
        onImport(data);
      } catch {
        alert("Invalid JSON file");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  return (
    <header className="cfg-header">
      <div className="cfg-header-left">
        <BookOpen size={20} />
        <span className="cfg-header-title">RAG Configurator</span>
      </div>
      <div className="cfg-header-right">
        <button className="cfg-header-btn" onClick={handleImportClick}>
          <Upload size={14} />
          Import
        </button>
        <button className="cfg-header-btn" onClick={onExport}>
          <Download size={14} />
          Export
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".json"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
      </div>
    </header>
  );
}
