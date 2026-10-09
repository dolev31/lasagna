import React, { useState } from "react";
import { ChevronDown } from "lucide-react";

interface Props {
  title: string;
  description?: string;
  defaultExpanded?: boolean;
  children: React.ReactNode;
}

export default function FieldGroup({ title, description, defaultExpanded = true, children }: Props) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className="cfg-field-group">
      <div className="cfg-field-group-header" onClick={() => setExpanded(!expanded)}>
        <div>
          <div className="cfg-field-group-title">{title}</div>
          {description && <div className="cfg-field-group-desc">{description}</div>}
        </div>
        <ChevronDown
          size={16}
          className={`cfg-field-group-chevron ${expanded ? "" : "collapsed"}`}
        />
      </div>
      <div className={`cfg-field-group-body ${expanded ? "" : "collapsed"}`}>
        {children}
      </div>
    </div>
  );
}
