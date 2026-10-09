import React, { useEffect, useState } from "react";

interface ToastMessage {
  text: string;
  tone: "success" | "error";
}

let _show: (msg: ToastMessage) => void = () => {};

export function showToast(text: string, tone: "success" | "error" = "success") {
  _show({ text, tone });
}

export default function Toast() {
  const [msg, setMsg] = useState<ToastMessage | null>(null);

  useEffect(() => {
    _show = (m) => {
      setMsg(m);
      setTimeout(() => setMsg(null), 2500);
    };
  }, []);

  if (!msg) return null;

  return (
    <div className={`cfg-toast ${msg.tone}`}>
      {msg.text}
    </div>
  );
}
