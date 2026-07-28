export default function Feature({ text, title }) {
  return (
    <div className="feature">
      <span aria-hidden="true">+</span>
      <div>
        <strong>{title}</strong>
        <p>{text}</p>
      </div>
    </div>
  );
}
