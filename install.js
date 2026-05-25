// Postinstall: install the Python package so the CLI entry point is available
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const pkgDir = path.resolve(__dirname);

// Check if pip is available
function checkPython() {
  try {
    execSync("python --version", { stdio: "pipe" });
    return "python";
  } catch (e) {
    try {
      execSync("python3 --version", { stdio: "pipe" });
      return "python3";
    } catch (e2) {
      return null;
    }
  }
}

function main() {
  const python = checkPython();
  if (!python) {
    console.warn("[genagent] ⚠ Python 3.11+ is required but not found in PATH.");
    console.warn("[genagent] Please install Python first: https://www.python.org/downloads/");
    process.exit(0); // soft-fail — npm install should not crash
  }

  // Check if human-memory is already installed
  try {
    execSync(`${python} -c "import human_memory"`, { stdio: "pipe" });
    console.log("[genagent] ✓ human-memory (Python) already installed");
  } catch (e) {
    console.log("[genagent] Installing Python package...");
    try {
      execSync(`"${python}" -m pip install "${pkgDir}"`, {
        stdio: "inherit",
        cwd: pkgDir,
      });
      console.log("[genagent] ✓ Python package installed");
    } catch (e2) {
      console.warn("[genagent] ⚠ pip install failed. Try manually:");
      console.warn(`[genagent]   pip install "${pkgDir}"`);
      process.exit(0);
    }
  }

  // Show env var options
  console.log("[genagent] ℹ Env vars: HUMAN_MEMORY_DB_PATH, HUMAN_MEMORY_EPISODIC_CAPACITY, HUMAN_MEMORY_EMBEDDING_MODEL");
}

main();
