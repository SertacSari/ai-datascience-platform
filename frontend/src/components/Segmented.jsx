export default function Segmented({ onChange, value, values }) {
  return (
    <div className="segmented" role="group" aria-label="Time range">
      {values.map((item) => (
        <button
          aria-pressed={value === item}
          className={value === item ? "active" : ""}
          key={item}
          onClick={() => onChange(item)}
          type="button"
        >
          {item}
        </button>
      ))}
    </div>
  );
}
