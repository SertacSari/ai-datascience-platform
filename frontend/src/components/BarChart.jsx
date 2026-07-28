import { useEffect, useRef } from "react";
import { drawBars } from "../lib/charts";

export default function BarChart({ label }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const draw = () => drawBars(canvas);
    draw();
    const observer = new ResizeObserver(draw);
    if (canvas.parentElement) observer.observe(canvas.parentElement);
    const themeObserver = new MutationObserver(draw);
    themeObserver.observe(document.documentElement, { attributeFilter: ["data-theme"], attributes: true });
    return () => {
      observer.disconnect();
      themeObserver.disconnect();
    };
  }, []);

  return <canvas aria-label={label} className="bar-canvas" ref={ref} role="img" />;
}
