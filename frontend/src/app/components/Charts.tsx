"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from "recharts";

// Warna untuk setiap kelompok IPK (8 kelompok, 0-7)
const COLORS = [
  "#ef4444", // 0 - merah
  "#f97316", // 1 - oranye
  "#eab308", // 2 - kuning
  "#84cc16", // 3 - lime
  "#22c55e", // 4 - hijau
  "#14b8a6", // 5 - teal
  "#6366f1", // 6 - indigo
  "#8b5cf6", // 7 - ungu
];

const IPK_SHORT: Record<number, string> = {
  0: "< 1.00",
  1: "1.00-1.49",
  2: "1.50-1.99",
  3: "2.00-2.49",
  4: "2.50-2.99",
  5: "3.00-3.49",
  6: "3.50-3.74",
  7: "3.75-4.00",
};

// ─── Gauge Setengah Lingkaran ───
export function IpkGauge({ value, max = 7 }: { value: number; max?: number }) {
  const radius = 80;
  const cx = 100;
  const cy = 95;
  const startAngle = Math.PI;
  const endAngle = 0;
  const fraction = Math.min(value / max, 1);

  // Arc background
  const bgD = describeArc(cx, cy, radius, startAngle, endAngle);
  // Arc filled
  const fillEnd = startAngle - fraction * Math.PI;
  const fillD = describeArc(cx, cy, radius, startAngle, fillEnd);

  const color = COLORS[value] ?? "#6366f1";

  return (
    <svg width="200" height="120" viewBox="0 0 200 120">
      {/* background arc */}
      <path d={bgD} fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
      {/* filled arc */}
      <path d={fillD} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round" />
      {/* center value */}
      <text x={cx} y={cy - 10} textAnchor="middle" fontSize="28" fontWeight="700" fill="currentColor" fontFamily="Poppins, sans-serif">
        {value}
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontSize="10" fill="#94a3b8" fontFamily="Poppins, sans-serif">
        Kelompok IPK
      </text>
    </svg>
  );
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const x1 = cx + r * Math.cos(startAngle);
  const y1 = cy - r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(endAngle);
  const y2 = cy - r * Math.sin(endAngle);
  const largeArc = Math.abs(startAngle - endAngle) > Math.PI ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
}

// ─── Bar Chart Probabilitas ───
export function ProbabilityBarChart({
  probabilities,
}: {
  probabilities: Record<string, number>;
}) {
  const data = Object.entries(probabilities).map(([key, val]) => {
    const idx = parseInt(key.replace("Grade ", ""));
    return {
      name: IPK_SHORT[idx] ?? key,
      persen: parseFloat((val * 100).toFixed(1)),
      idx,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <XAxis
          dataKey="name"
          tick={{ fontSize: 10, fontFamily: "Poppins" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 10, fontFamily: "Poppins" }}
          axisLine={false}
          tickLine={false}
          unit="%"
          domain={[0, 100]}
        />
        <Tooltip
          contentStyle={{
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "Poppins",
            border: "1px solid #e2e8f0",
          }}
          formatter={(v) => [`${v}%`, "Probabilitas"]}
        />
        <Bar dataKey="persen" radius={[4, 4, 0, 0]} animationDuration={600}>
          {data.map((entry) => (
            <Cell key={entry.name} fill={COLORS[entry.idx] ?? "#6366f1"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Donut Chart Distribusi Audit ───
export function AuditDonutChart({
  summary,
  total,
}: {
  summary: Record<string, number>;
  total: number;
}) {
  const data = Object.entries(summary).map(([key, val]) => {
    const idx = parseInt(key.replace("Grade ", ""));
    return {
      name: IPK_SHORT[idx] ?? key,
      value: val,
      idx,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={55}
          outerRadius={85}
          paddingAngle={3}
          dataKey="value"
          animationDuration={600}
          label={({ name, percent }) =>
            `${name} (${((percent ?? 0) * 100).toFixed(0)}%)`
          }
          labelLine={false}
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={COLORS[entry.idx] ?? "#6366f1"} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "Poppins",
            border: "1px solid #e2e8f0",
          }}
          formatter={(v) => [`${v} mahasiswa`, "Jumlah"]}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
