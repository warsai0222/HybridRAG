#!/bin/bash
# Waits for Groq daily TPD reset (midnight UTC) then runs the eval automatically.
# Run this before bed: bash run_eval_tonight.sh
# Results saved to: data/eval/eval_results_YYYYMMDD.log

echo "⏳ Calculating wait time until Groq reset (midnight UTC)..."

WAIT=$(python3 -c "
import datetime, time
now = datetime.datetime.utcnow()
midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
secs = int((midnight - now).total_seconds())
print(secs)
")

HOURS=$((WAIT / 3600))
MINS=$(( (WAIT % 3600) / 60 ))
echo "🕐 Waiting ${HOURS}h ${MINS}m (midnight UTC + 5min buffer)..."
echo "   Go to sleep — this will run on its own."
echo ""

sleep $WAIT

echo "🚀 Groq reset confirmed. Running HybridRAG eval..."
cd "$(dirname "$0")"

python -m src.eval.evaluate --data data/eval/eval_large.jsonl 2>&1 | tee "data/eval/eval_results_$(date +%Y%m%d).log"

echo ""
echo "✅ Eval complete. Results saved to data/eval/eval_results_$(date +%Y%m%d).log"
echo "   Check accuracy, macro F1, and calibration error in the morning."
