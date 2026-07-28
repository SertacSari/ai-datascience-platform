export const rangePoints = { "1W": 7, "1M": 30, "3M": 90, "1Y": 52 };

function prepCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, height: rect.height, width: rect.width };
}

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function alphaColor(color, alpha) {
  const hex = color.replace("#", "").trim();
  if (!/^[\da-f]{6}$/i.test(hex)) return color;
  const parts = [0, 2, 4].map((start) => parseInt(hex.slice(start, start + 2), 16));
  return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
}

export function drawLine(canvas, range) {
  const { ctx, height, width } = prepCanvas(canvas);
  const primary = token("--primary");
  const border = token("--border");
  const muted = token("--muted");
  const card = token("--card");
  const count = rangePoints[range];
  const data = Array.from({ length: count }, (_, index) => {
    return 62 + Math.sin(index / 3.4) * 16 + Math.cos(index / 8) * 9 + index * 0.18;
  });
  const min = Math.min(...data) - 8;
  const max = Math.max(...data) + 8;
  const pad = 26;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = card;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = pad + ((height - pad * 2) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  const xy = (value, index) => [
    pad + (index / (data.length - 1)) * (width - pad * 2),
    height - pad - ((value - min) / (max - min)) * (height - pad * 2)
  ];

  ctx.beginPath();
  data.forEach((value, index) => {
    const [x, y] = xy(value, index);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(width - pad, height - pad);
  ctx.lineTo(pad, height - pad);
  ctx.closePath();
  ctx.fillStyle = alphaColor(primary, 0.1);
  ctx.fill();

  ctx.beginPath();
  data.forEach((value, index) => {
    const [x, y] = xy(value, index);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = primary;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  const last = xy(data[data.length - 1], data.length - 1);
  ctx.fillStyle = primary;
  ctx.beginPath();
  ctx.arc(last[0], last[1], 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(last[0], last[1]);
  ctx.lineTo(width - 10, Math.max(pad, last[1] - 18));
  ctx.strokeStyle = muted;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
}

export function drawBars(canvas) {
  const { ctx, height, width } = prepCanvas(canvas);
  const primary = token("--primary");
  const border = token("--border");
  const muted = token("--muted");
  const values = [28, 35, 32, 44, 48, 46, 58, 62, 59, 68, 72, 81];
  const pad = 18;
  const gap = 6;
  const barWidth = (width - pad * 2) / values.length - gap;

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i += 1) {
    const y = pad + ((height - pad * 2) / 3) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  values.forEach((value, index) => {
    const x = pad + index * (barWidth + gap);
    const barHeight = (value / 90) * (height - pad * 2);
    ctx.globalAlpha = index > 8 ? 1 : 0.38;
    ctx.fillStyle = index > 8 ? primary : muted;
    ctx.fillRect(x, height - pad - barHeight, barWidth, barHeight);
  });
  ctx.globalAlpha = 1;
}
