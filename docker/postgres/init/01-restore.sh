#!/usr/bin/env sh
set -euo pipefail

DUMP_PATH="/import/industrial_db_full_latest.dump"
DB="${POSTGRES_DB:-industrial_db}"
USER="${POSTGRES_USER:-postgres}"

echo "[init] Preparing to restore dump into database '${DB}'"

if [ ! -f "$DUMP_PATH" ]; then
  echo "[init] Dump file not found at $DUMP_PATH, skipping restore."
  exit 0
fi

# Try to detect if the dump is a custom format (pg_dump -Fc)
if pg_restore -l "$DUMP_PATH" > /dev/null 2>&1; then
  echo "[init] Detected custom-format dump; restoring via pg_restore..."
  pg_restore --verbose --clean --if-exists --no-owner -U "$USER" -d "$DB" "$DUMP_PATH"
else
  echo "[init] Dump not recognized as custom format; applying with psql..."
  psql -U "$USER" -d "$DB" -v ON_ERROR_STOP=1 -f "$DUMP_PATH"
fi

echo "[init] Database restore completed."