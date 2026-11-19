#!/usr/bin/env bash
set -u
B=/root/autodl-tmp/fast30_final
D=$B/RQ4_EXPERIMENT_HUB_20261014/deep3_extension_20261015
V=$B/.venvs/tf220_clean_20261014/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
cd "$D/runtime" || exit 90
mkdir -p "$D/logs/d100"
for i in $(seq -w 1 30); do
  free=$(df -B1 /root/autodl-tmp |awk 'NR==2{print $4}')
  if ((free < 5*1024*1024*1024)); then echo "STOP_DISK_GUARD seed=$i free=$free"; exit 75; fi
  name="s4_two_chains_deep3_n5000_d100_s$i"
  status="$D/runs/$name/S$i/synthetic_status.json"
  if test -f "$status";then echo "BLOCKED_EXISTING $name";exit 76;fi
  echo "START $name $(date -Is)"
  "$V" -u synthetic_runner.py --contains "$name" --limit 1 --workers 2 > "$D/logs/d100/$name.log" 2>&1
  rc=$?
  "$V" - "$status" "$rc" <<'PY'
import json,sys,pathlib
s=pathlib.Path(sys.argv[1]);rc=int(sys.argv[2])
print("CASE_RESULT",s.parent.parent.name,"rc",rc,"status",json.load(s.open()).get("status") if s.exists() else "missing",flush=True)
PY
  if ((rc != 0)) || ! test -f "$D/runs/$name/S$i/record.json";then echo "STOP_FAILURE $name";exit 77;fi
done
echo "S4_D100_30_COMPLETE"
