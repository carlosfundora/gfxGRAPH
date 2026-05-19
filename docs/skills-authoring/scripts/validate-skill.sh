#!/bin/bash
set -euo pipefail

SKILL_DIR=${1:-}

if [ -z "$SKILL_DIR" ]; then
  echo "Usage: $0 <path-to-skill-dir>"
  exit 1
fi

if [ ! -d "$SKILL_DIR" ]; then
  echo "Error: Directory '$SKILL_DIR' does not exist."
  exit 1
fi

if [ ! -f "$SKILL_DIR/SKILL.md" ]; then
  echo "Error: Missing SKILL.md in '$SKILL_DIR'"
  exit 1
fi

echo "Validating SKILL.md structure..."

if ! grep -q "^---$" "$SKILL_DIR/SKILL.md"; then
  echo "Error: Missing YAML frontmatter in SKILL.md"
  exit 1
fi

if ! grep -q "^name:[[:space:]]\\+[^[:space:]].*$" "$SKILL_DIR/SKILL.md"; then
  echo "Error: Missing name field in SKILL.md frontmatter"
  exit 1
fi

if ! grep -q "^description:[[:space:]]\\+[^[:space:]].*$" "$SKILL_DIR/SKILL.md"; then
  echo "Error: Missing description field in SKILL.md frontmatter"
  exit 1
fi

if ! grep -q "^# " "$SKILL_DIR/SKILL.md"; then
  echo "Error: Missing H1 title in SKILL.md"
  exit 1
fi

echo "Skill '$SKILL_DIR' passed validation."
