export default function StatCard({ delta, label, tone, value }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <em className={tone}>{delta}</em>
    </div>
  );
}
