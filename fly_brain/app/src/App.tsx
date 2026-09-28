import { useEffect, useState } from "react";
import { loadBar, loadWalk, MOOD_FACE, MOOD_WORD, type BarData, type Drink, type WalkData } from "./data";
import { BarScene } from "./scene/BarScene";
import { BrainView } from "./brain/BrainView";
import { configureTour, giveDrink, setSpeed, setView, startTour, useTour, useTourClock } from "./store";

const nf = new Intl.NumberFormat("en-US");

const REASON: Record<Drink["mood"], string> = {
  love: "Of everything on the menu, this is the one he’d order.",
  like: "He likes it, but not as much as the one he ordered.",
  meh: "He’s not sure. He’d probably pick something else.",
  yuck: "Too bitter for a fly. He wouldn’t order it.",
};

function Status({ data, walk }: { data: BarData; walk: WalkData | null }) {
  const key = useTour((s) => `${s.phase}|${s.current ?? ""}|${s.target ?? ""}|${s.finale}|${s.tasted.length}`);
  const [phase, cur, tgt, finale, tastedCount] = key.split("|");
  const at = cur !== "" ? data.drinks[Number(cur)] : null;
  const to = tgt !== "" ? data.drinks[Number(tgt)] : null;

  if (phase === "settled" && at) {
    return (
      <div className="status chosen">
        <p className="eyebrow">He’d order</p>
        <h2 className="status-title">
          <span aria-hidden="true">{at.emoji}</span> {at.name}
        </h2>
        <p className="status-text">{REASON.love}</p>
      </div>
    );
  }
  let eyebrow = "At the bar";
  let title = "Wandering along the counter";
  let text = walk
    ? "The neurons in his brain that move his legs decide where he goes and when he turns. Press “Start the tasting” to have him try your drinks."
    : "He’s waiting for you to give him drinks. Press “Start the tasting” to have him try them one by one.";
  if (phase === "flying" && to) {
    eyebrow = finale === "true" ? "He’s decided" : `Drink ${Number(tastedCount) + 1} of ${data.drinks.length}`;
    title = finale === "true" ? `Heading back to the ${to.name}` : `Flying to the ${to.name}`;
    text = finale === "true" ? "He’s tried them all and knows what he wants." : "He lands on the rim of the glass.";
  } else if (phase === "tasting" && at) {
    eyebrow = "Tasting";
    title = at.name;
    text = "He steps into the drink, extends his proboscis, and his brain decides.";
  } else if ((phase === "reacting" || phase === "idle") && at && data.provisional) {
    eyebrow = "Tasted";
    title = at.name;
    text =
      Number(tastedCount) === data.drinks.length
        ? "He’s tried them all. His order will be ready once the computation finishes."
        : "His brain reacted. His verdict will show once the computation finishes.";
  } else if ((phase === "reacting" || phase === "idle") && at) {
    eyebrow = "His verdict";
    title = `${MOOD_FACE[at.mood]} ${MOOD_WORD[at.mood]}`;
    text = `${at.name}: ${REASON[at.mood]}`;
  }
  return (
    <div className="status">
      <p className="eyebrow">{eyebrow}</p>
      <h2 className="status-title">{title}</h2>
      <p className="status-text">{text}</p>
    </div>
  );
}

function ViewHint() {
  const free = useTour((s) => s.view === "free");
  if (!free) return null;
  return <p className="view-hint">Drag to rotate · scroll to zoom</p>;
}

function Controls({ data }: { data: BarData }) {
  const speed = useTour((s) => s.speed);
  const view = useTour((s) => s.view);
  const running = useTour((s) => s.phase !== "idle" || s.current !== null);
  return (
    <div className="controls">
      <button type="button" className="btn primary" onClick={startTour}>
        {running ? "Start over" : "Start the tasting"}
      </button>
      <button type="button" className="btn" onClick={() => setSpeed(speed === 1 ? 2.5 : 1)}>
        {speed === 1 ? "Faster" : "Normal speed"}
      </button>
      <button type="button" className="btn" onClick={() => setView(view === "follow" ? "free" : "follow")}>
        {view === "follow" ? "Whole bar" : "Follow the fly"}
      </button>
      <div className="picker" role="group" aria-label="Give him a drink">
        <span className="picker-label">Give him:</span>
        {data.drinks.map((d, i) => (
          <button key={d.id} type="button" className="chip" onClick={() => giveDrink(i)}>
            <span aria-hidden="true">{d.emoji}</span> {d.name}
          </button>
        ))}
      </div>
    </div>
  );
}

function Ranking({ data }: { data: BarData }) {
  const tastedKey = useTour((s) => [...s.tasted].sort((a, b) => a - b).join(","));
  const settled = useTour((s) => s.phase === "settled");
  const tasted = new Set(tastedKey ? tastedKey.split(",").map(Number) : []);
  if (data.provisional) {
    return (
      <div className="card ranking">
        <div className="card-head">
          <h3>What he’d order</h3>
          <span className="card-sub">still computing</span>
        </div>
        <p className="empty">
          The fly hasn’t decided yet. His brain is comparing how readily he’d drink each one.
        </p>
        <ol className="rank-list">
          {data.drinks.map((d, i) => (
            <li key={d.id} className={tasted.has(i) ? "" : "unknown"}>
              <span className="rank-pos">·</span>
              <span className="rank-name">
                <span aria-hidden="true">{d.emoji}</span> {d.name}
              </span>
              <span className="rank-bar" aria-hidden="true" />
              <span className="rank-mood">{tasted.has(i) ? "tasted" : "—"}</span>
            </li>
          ))}
        </ol>
      </div>
    );
  }
  const rows = [...data.drinks].map((d, i) => ({ d, i })).sort((a, b) => a.d.rank - b.d.rank);
  return (
    <div className="card ranking">
      <div className="card-head">
        <h3>What he’d order</h3>
        <span className="card-sub">{tasted.size} of {data.drinks.length} tasted</span>
      </div>
      <ol className="rank-list">
        {rows.map(({ d, i }) => {
          const known = tasted.has(i) || settled;
          return (
            <li key={d.id} className={`${known ? "" : "unknown"} ${settled && d.rank === 1 ? "winner" : ""}`}>
              <span className="rank-pos">{known ? d.rank : "·"}</span>
              <span className="rank-name">
                <span aria-hidden="true">{d.emoji}</span> {d.name}
              </span>
              <span className="rank-bar" aria-hidden="true">
                <span className={`rank-fill ${d.mood}`} style={{ width: known ? `${Math.max(4, d.likes)}%` : "0%" }} />
              </span>
              <span className={`rank-mood ${known ? d.mood : ""}`}>{known ? MOOD_WORD[d.mood] : "—"}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function BrainRegions({ data }: { data: BarData }) {
  const cur = useTour((s) => s.current ?? -1);
  const d = data.drinks[cur];
  const cloud = d?.cloud;
  if (!cloud || !cloud.regions.length) {
    return (
      <p className="empty">
        {d ? `We haven’t recorded which neurons light up for the ${d.name} yet.` : "When he tastes a drink, this shows which parts of his brain light up."}
      </p>
    );
  }
  const max = Math.max(...cloud.regions.map((r) => r.lit));
  return (
    <div className="regions">
      <p className="regions-total">
        <span aria-hidden="true">{d.emoji}</span> With the {d.name}, <b>{nf.format(cloud.lit)}</b> neurons light up
      </p>
      <ul>
        {cloud.regions.map((r) => (
          <li key={r.name}>
            <span className="region-name">{r.name}</span>
            <span className="region-bar" aria-hidden="true">
              <span style={{ width: `${Math.max(3, (r.lit / max) * 100)}%` }} />
            </span>
            <span className="region-n">{nf.format(r.lit)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TasteCard({ data }: { data: BarData }) {
  const cur = useTour((s) => s.current ?? s.target ?? -1);
  const d = data.drinks[cur];
  const rows: [string, number, string][] = d
    ? [["Sweetness", d.taste.sweet, "like"], ["Bitterness", d.taste.bitter, "yuck"], ["Carbonation", d.taste.fizz, "fizz"], ["Saltiness", d.taste.salt, "salt"]]
    : [];
  return (
    <div className="card taste">
      <div className="card-head">
        <h3>What he tastes</h3>
        <span className="card-sub">{d ? `${d.emoji} ${d.name}` : "pick a drink"}</span>
      </div>
      {d ? (
        <>
          <div className="taste-rows">
            {rows.map(([label, v, cls]) => (
              <div key={label} className="taste-row">
                <span>{label}</span>
                <span className="taste-track">
                  <span className={`taste-fill ${cls}`} style={{ width: `${Math.round(v * 100)}%` }} />
                </span>
              </div>
            ))}
          </div>
          <dl className="recipe">
            {d.recipe.map((r) => (
              <div key={r.label}>
                <dt>{r.label}</dt>
                <dd>{r.value}</dd>
              </div>
            ))}
          </dl>
        </>
      ) : (
        <p className="empty">When the fly tastes a drink, this shows what he senses on his tongue and legs.</p>
      )}
    </div>
  );
}

export default function App() {
  const [data, setData] = useState<BarData | null>(null);
  const [walk, setWalk] = useState<WalkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useTourClock();

  useEffect(() => {
    loadBar()
      .then((d) => {
        setData(d);
        configureTour(d.drinks.length, d.drinks.findIndex((x) => x.id === d.winner), !d.provisional);
      })
      .catch((e: Error) => setError(e.message));
    loadWalk().then(setWalk);
  }, []);

  if (error) return <div className="fatal">{error}</div>;
  if (!data) return <div className="fatal">Opening the bar…</div>;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">🪰</span>
          <div>
            <h1>Fly Bar</h1>
            <p>A fly with a real brain picks a drink</p>
          </div>
        </div>
        <div className="facts">
          <div>
            <span>Brain</span>
            <b>{nf.format(data.neurons)} neurons</b>
          </div>
          <div>
            <span>Map</span>
            <b>Google · Janelia</b>
          </div>
          <div>
            <span>Menu</span>
            <b>{data.drinks.length} drinks</b>
          </div>
        </div>
      </header>

      <section className="stage">
        <BarScene data={data} walk={walk} />
        <div className="stage-overlay">
          <Status data={data} walk={walk} />
        </div>
        <ViewHint />
      </section>

      <Controls data={data} />

      <section className="lower">
        <div className="card brain">
          <div className="card-head">
            <h3>His brain right now</h3>
            <span className="card-sub">{nf.format(data.points)} neurons, each in its place · drag to rotate</span>
          </div>
          <BrainView data={data} />
          <BrainRegions data={data} />
        </div>
        <div className="side">
          <Ranking data={data} />
          <TasteCard data={data} />
        </div>
      </section>

      <footer className="foot">
        {data.provisional && <p className="warn">His order is still being computed.</p>}
        <p>
          The brain is the complete map of a male fruit fly (Google & Janelia, 2026). For each drink we ran a
          simulation of the whole brain on a computer, and what you see here is its replay: which neurons lit up, how
          much he wanted to drink, and how bitter it tasted to him. To a fly, alcohol and acidity taste bitter, so he
          orders the cocktail that tastes least bitter for how sweet it is.
        </p>
      </footer>
    </div>
  );
}
