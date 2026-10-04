cd /home/oem/choihyunsu/BreastCancerDetectionSystem
DEST="/mnt/external_ssd/BreastCancer Datasets/EMBED"
for pass in 1 2 3 4 5 6; do
  n=$(find "$DEST" -name "*.dcm" 2>/dev/null | wc -l)
  echo "[$(date +%T)] pass $pass start: $n/14000"
  [ "$n" -ge 14000 ] && { echo COMPLETE; break; }
  ~/.local/bin/rclone copy embed: "$DEST" --files-from embed_subset.txt \
    --no-traverse --tpslimit 10 --transfers 4 --retries 20 --low-level-retries 20 \
    --drive-acknowledge-abuse --stats=60s --stats-one-line -v >> embed_pull_resume.log 2>&1
  echo "[$(date +%T)] pass $pass done"; sleep 120
done
echo "[$(date +%T)] FINISHER DONE: $(find "$DEST" -name '*.dcm' 2>/dev/null | wc -l)/14000"
