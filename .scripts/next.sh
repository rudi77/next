#!/usr/bin/env bash
#
# next.sh — install / run / stop the trainpipe ("next") service.
#
# Target: the Linux deploy box (4×GPU). Creates a project-local virtualenv,
# installs trainpipe in editable mode, and manages the uvicorn server as a
# backgrounded process tracked by a PID file.
#
# Usage:
#   ./.scripts/next.sh install      # create .venv + pip install -e ".[training,dev]" + seed .env
#   ./.scripts/next.sh start        # launch the API server in the background
#   ./.scripts/next.sh stop         # stop it
#   ./.scripts/next.sh restart      # stop then start
#   ./.scripts/next.sh status       # PID + HTTP health probe
#   ./.scripts/next.sh logs [-f]    # tail the server log (-f to follow)
#
# Knobs (env vars):
#   PYTHON          python interpreter for venv creation   (default: python3)
#   TRAINPIPE_EXTRAS  pip extras to install                (default: training,dev)
#
set -euo pipefail

# --- locate project root (parent of this .scripts/ dir) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_DIR="${PROJECT_ROOT}/.venv"
RUN_DIR="${PROJECT_ROOT}/.run"
PIDFILE="${RUN_DIR}/trainpipe.pid"
LOGFILE="${RUN_DIR}/trainpipe.log"

PYTHON="${PYTHON:-python3}"
TRAINPIPE_EXTRAS="${TRAINPIPE_EXTRAS:-training,dev}"

# --- helpers ---------------------------------------------------------------
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Read TRAINPIPE_HOST / TRAINPIPE_PORT from the environment, else .env, else default.
config_value() {
    local key="$1" default="$2"
    if [[ -n "${!key:-}" ]]; then
        printf '%s' "${!key}"
    elif [[ -f "${PROJECT_ROOT}/.env" ]]; then
        local v
        v="$(grep -E "^${key}=" "${PROJECT_ROOT}/.env" | tail -n1 | cut -d= -f2- || true)"
        printf '%s' "${v:-$default}"
    else
        printf '%s' "$default"
    fi
}

health_url() {
    local host port
    host="$(config_value TRAINPIPE_HOST 0.0.0.0)"
    port="$(config_value TRAINPIPE_PORT 8080)"
    # 0.0.0.0 isn't a connectable address from a client — probe loopback.
    [[ "$host" == "0.0.0.0" || -z "$host" ]] && host="127.0.0.1"
    printf 'http://%s:%s/health' "$host" "$port"
}

is_running() {
    [[ -f "${PIDFILE}" ]] || return 1
    local pid
    pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# --- commands --------------------------------------------------------------
cmd_install() {
    command -v "${PYTHON}" >/dev/null 2>&1 || die "'${PYTHON}' not found; set PYTHON=/path/to/python3"

    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Creating virtualenv at ${VENV_DIR}"
        "${PYTHON}" -m venv "${VENV_DIR}"
    else
        log "Reusing existing virtualenv at ${VENV_DIR}"
    fi

    log "Upgrading pip"
    "${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null

    log "Installing trainpipe (editable) with extras: [${TRAINPIPE_EXTRAS}]"
    "${VENV_DIR}/bin/pip" install -e ".[${TRAINPIPE_EXTRAS}]"

    if [[ ! -f "${PROJECT_ROOT}/.env" && -f "${PROJECT_ROOT}/.env.example" ]]; then
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
        warn "Seeded .env from .env.example — edit TRAINPIPE_API_KEY before exposing the server."
    fi

    log "Install complete."
    echo "  • GPU training also needs a CUDA torch build: ./.scripts/install-torch-cu128.sh"
    echo "  • MLflow tracking server:                      docker compose up -d"
    echo "  • Start the API:                               ./.scripts/next.sh start"
}

cmd_start() {
    [[ -x "${VENV_DIR}/bin/trainpipe" ]] || die "not installed — run './.scripts/next.sh install' first"
    if is_running; then
        log "Already running (pid $(cat "${PIDFILE}"))."
        return 0
    fi

    mkdir -p "${RUN_DIR}"
    log "Starting trainpipe…"
    # nohup + setsid so the server survives the shell that launched it.
    nohup "${VENV_DIR}/bin/trainpipe" >>"${LOGFILE}" 2>&1 &
    echo $! >"${PIDFILE}"

    # Give uvicorn a moment, then confirm it's actually up.
    local url; url="$(health_url)"
    for _ in $(seq 1 20); do
        if is_running && curl -fsS "${url}" >/dev/null 2>&1; then
            log "Up (pid $(cat "${PIDFILE}")) — health OK at ${url}"
            return 0
        fi
        sleep 0.5
    done

    if is_running; then
        warn "Process is alive (pid $(cat "${PIDFILE}")) but ${url} didn't answer yet."
        warn "Check logs: ./.scripts/next.sh logs"
    else
        rm -f "${PIDFILE}"
        die "Server exited during startup. Last log lines:
$(tail -n 20 "${LOGFILE}" 2>/dev/null || true)"
    fi
}

cmd_stop() {
    if ! is_running; then
        log "Not running."
        rm -f "${PIDFILE}"
        return 0
    fi
    local pid; pid="$(cat "${PIDFILE}")"
    log "Stopping pid ${pid}…"
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "${pid}" 2>/dev/null || { rm -f "${PIDFILE}"; log "Stopped."; return 0; }
        sleep 0.5
    done
    warn "Didn't exit gracefully — sending SIGKILL."
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${PIDFILE}"
    log "Killed."
}

cmd_restart() { cmd_stop; cmd_start; }

cmd_status() {
    local url; url="$(health_url)"
    if is_running; then
        log "Running (pid $(cat "${PIDFILE}"))."
    else
        log "Not running."
    fi
    if curl -fsS "${url}" >/dev/null 2>&1; then
        echo "  health: OK   (${url})"
    else
        echo "  health: DOWN (${url})"
    fi
}

cmd_logs() {
    [[ -f "${LOGFILE}" ]] || die "no log file yet at ${LOGFILE}"
    if [[ "${1:-}" == "-f" ]]; then
        tail -n 100 -f "${LOGFILE}"
    else
        tail -n 100 "${LOGFILE}"
    fi
}

usage() {
    # Print the leading comment block (skipping the shebang), stripping "# ".
    awk 'NR>2 && /^#/ {sub(/^# ?/, ""); print; next} NR>2 {exit}' "${BASH_SOURCE[0]}"
}

# --- dispatch --------------------------------------------------------------
case "${1:-}" in
    install) cmd_install ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    status)  cmd_status ;;
    logs)    shift; cmd_logs "${@:-}" ;;
    ""|-h|--help|help) usage ;;
    *) die "unknown command '${1}'. Run './.scripts/next.sh help'." ;;
esac
