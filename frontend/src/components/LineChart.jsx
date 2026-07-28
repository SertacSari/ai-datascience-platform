import { useEffect, useRef } from "react";
import { drawLine } from "../lib/charts";

export default function LineChart({ label, range }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const draw = () => drawLine(canvas, range);
    draw();
    const observer = new ResizeObserver(draw);
    if (canvas.parentElement) observer.observe(canvas.parentElement);
    const themeObserver = new MutationObserver(draw);
    themeObserver.observe(document.documentElement, { attributeFilter: ["data-theme"], attributes: true });
    return () => {
      observer.disconnect();
      themeObserver.disconnect();
    };
  }, [range]);

  return <canvas aria-label={label} className="chart-canvas" ref={ref} role="img" />;
}
