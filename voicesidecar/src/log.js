// Minimal levelled logger (LOG_LEVEL: debug | info | warn | error).

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };
const threshold = LEVELS[(process.env.LOG_LEVEL || "info").toLowerCase()] ?? LEVELS.info;

function write(level, message) {
  if (LEVELS[level] < threshold) return;
  const line = `${new Date().toISOString()} ${level.toUpperCase().padEnd(5)} ${message}`;
  (level === "error" || level === "warn" ? console.error : console.log)(line);
}

export const log = {
  debug: (message) => write("debug", message),
  info: (message) => write("info", message),
  warn: (message) => write("warn", message),
  error: (message) => write("error", message),
};
