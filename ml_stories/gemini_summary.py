"""
Gemini summaries for operations staff (optional layer on top of the ML outputs).

What it does: reads output CSVs already produced by the ML stories (US-06 anomalies,
US-05 burst risk, US-12 tanker schedule, US-01 forecast) and asks Gemini to write a
short plain-language briefing for a given city. Gemini does NOT make predictions; it only
summarises numbers the models already produced.

Setup:  pip install google-genai
        set GEMINI_API_KEY=<your key>          (PowerShell: $env:GEMINI_API_KEY="...")
        optional: set GEMINI_MODEL=<model name your key supports>
Usage:  python ml_stories/gemini_summary.py --city 3            (calls the API)
        python ml_stories/gemini_summary.py --city 3 --dry-run  (prints the prompt only)
"""
import argparse, os
import pandas as pd

def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def build_prompt(city_id):
    parts = []
    an = load("output/us06_supply_anomalies.csv")
    if an is not None:
        a = an[an.city_id == city_id]
        parts.append(f"Supply anomalies flagged: {len(a)} ward-days; "
                     f"lowest supply efficiency {a['supply_efficiency_pct'].min():.1f}%" if len(a) else "No supply anomalies flagged.")
    br = load("output/us05_pipeline_burst_risk.csv")
    if br is not None:
        b = br[br.city_id == city_id]
        parts.append("Burst-risk wards by label: " + b["risk_label"].value_counts().to_dict().__str__())
    tk = load("output/us12_tanker_schedule.csv")
    if tk is not None:
        t = tk[tk.city_id == city_id]
        parts.append(f"Wards flagged for tanker supplement next week: {int(t['tanker_needed'].sum())}; "
                     f"recommended tankers: {int(t.loc[t.tanker_needed == 1, 'recommended_tankers'].sum())}")
    fc = load("output/us01_12week_city_forecast.csv")
    if fc is not None and (fc.city_id == city_id).any():
        f = fc[fc.city_id == city_id].iloc[0]
        parts.append(f"Forecast demand (MLD) week 1: {f['week_1']}, week 12: {f['week_12']}")
    facts = "\n".join(f"- {p}" for p in parts) or "- (no model outputs found)"
    return ("You are writing a short briefing for a municipal water operations officer. "
            "Use ONLY the facts below, do not invent numbers, and note that the data are synthetic "
            f"and the model outputs are preliminary.\n\nCity id: {city_id}\n{facts}\n\n"
            "Write 4 to 6 sentences with the most important action first.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--city", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true"); args = ap.parse_args()
    prompt = build_prompt(args.city)
    if args.dry_run:
        print(prompt)
    else:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        text = client.models.generate_content(model=model, contents=prompt).text
        os.makedirs("output", exist_ok=True)
        open(f"output/gemini_summary_city_{args.city}.txt", "w", encoding="utf-8").write(text)
        print(text)