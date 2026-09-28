#!/usr/bin/env bash
# Assemble the static demo site into site/dist (Vercel runs this, see vercel.json).
#   /            site/index.html
#   /bar/        Fly Bar web app (fly_brain/app, built with vite)
#   /classic     Fly Bar single-page version (fly_brain/ui/fly_bar.html)
#   /dashboard/  Fly Brain Live (fly_brain/dashboard)
#   /ride/       the fly rides a Tarmac SL9 (fly_brain/results/ride_*.html)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=site/dist
rm -rf "$OUT"
mkdir -p "$OUT/dashboard/data" "$OUT/ride"

cp site/index.html "$OUT/"

(cd fly_brain/app && npm ci --no-audit --no-fund && npx vite build --base /bar/ --outDir ../../$OUT/bar --emptyOutDir)

cp fly_brain/ui/fly_bar.html "$OUT/classic.html"

cp fly_brain/dashboard/index.html "$OUT/dashboard/"
cp fly_brain/dashboard/data/*.json fly_brain/dashboard/data/*.png fly_brain/dashboard/data/*.js "$OUT/dashboard/data/"

R=fly_brain/results
cp $R/ride_3d.html          "$OUT/ride/index.html"   # brain steers: best learned decoder
cp $R/ride_view.html        "$OUT/ride/charts.html"
cp $R/ride_oracle_3d.html   "$OUT/ride/oracle.html"  # PD rider, no brain
cp $R/ride_oracle_view.html "$OUT/ride/oracle-charts.html"
cp $R/ride_pilot_3d.html    "$OUT/ride/open-loop.html"  # brain in the loop, no steering
cp $R/ride_pilot_view.html  "$OUT/ride/open-loop-charts.html"

du -sh "$OUT"
