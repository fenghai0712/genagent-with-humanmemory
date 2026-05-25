#!/usr/bin/env node
// Bridge: npm global bin → Python memory-agent CLI
const { execSync, spawn } = require("child_process");
const path = require("path");

// Try the installed CLI entry point first (created by pip install)
function findPythonCLI() {
  const candidates = [
    "memory-agent",                                 // on PATH (Windows: memory-agent.exe)
    "python -m human_memory.agent",                 // fallback: run as module
    "python3 -m human_memory.agent",                // macOS/Linux fallback
  ];

  for (const cmd of candidates) {
    try {
      const exe = cmd.split(" ")[0];
      const args = cmd.split(" ").slice(1);
      execSync(`${exe} --version`, { stdio: "pipe" });
      return { exe, args };
    } catch (e) {
      // try next
    }
  }

  // Last resort: try python -m directly
  return { exe: "python", args: ["-m", "human_memory.agent"] };
}

const { exe, args } = findPythonCLI();
const child = spawn(exe, args, {
  stdio: "inherit",
  shell: true,
});

child.on("exit", (code) => {
  process.exit(code || 0);
});
