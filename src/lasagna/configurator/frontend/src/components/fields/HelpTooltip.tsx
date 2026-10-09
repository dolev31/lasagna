import React, { useState } from "react";

interface Props {
  text: string;
}

export default function HelpTooltip({ text }: Props) {
  const [show, setShow] = useState(false);

  return (
    <span
      className="cfg-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <span className="cfg-help-icon">?</span>
      {show && <div className="cfg-help-tooltip">{text}</div>}
    </span>
  );
}
