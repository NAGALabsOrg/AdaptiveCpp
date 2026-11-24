# Report the names of the summary's primitive table and graph section that a program does not use.
#
#   nu naga/tasks/primitive-coverage.nu <program.cpp> [--summary <summary.md>]
#
# Exit code 1 and one absent name per line when something is missing.

const default_summary = "docs/naga-internal/brainstorm/language-pass/summary.md"
const name_pattern = '(?<name>(?:kh|sg|unit|item|g)(?:\.|::)[A-Za-z_]\w*(?:<sg::segmented>)?)'
const extra_names = ["sg::fill"]

def section [text: string, heading: string] {
  let ls = ($text | lines)
  let start = ($ls | enumerate | where {|e| $e.item | str starts-with $heading } | first | get index)
  let rest = ($ls | skip ($start + 1))
  let stops = ($rest | enumerate | where {|e| $e.item | str starts-with "## " } | get index)
  if ($stops | is-empty) { $rest } else { $rest | first ($stops | first) }
}

# The first cell of every row of the primitive table: every kh., sg::, unit. and item. name, and `.count`.
export def table-names [summary: string] {
  let rows = (section $summary "## The language" | where {|l| $l | str starts-with "| " } | skip 1)
  $rows | each {|row|
    let cell = ($row | split row '|' | get 1)
    let named = ($cell | parse --regex $name_pattern | get name)
    let dotted = ($cell | parse --regex '(?:^|\s)\.(?<name>[A-Za-z_]\w*)\(' | get name | each {|n| $".($n)" })
    $named | append $dotted
  } | flatten | uniq
}

# The names of the graph section: its code and its bullets.
export def graph-names [summary: string] {
  section $summary "## The graph around a kernel" | each {|l| $l | parse --regex $name_pattern | get name } | flatten | uniq
}

export def wanted-names [summary: string] {
  (table-names $summary) | append (graph-names $summary) | append $extra_names | uniq
}

def present [program: string, name: string] {
  let literal = ($name | str replace --all '.' '[.]' | str replace --all '<' '[<]' | str replace --all '>' '[>]')
  $program =~ ($literal + '(?:[^A-Za-z0-9_]|$)')
}

export def absent-names [program: string, summary: string] {
  wanted-names $summary | where {|n| not (present $program $n) }
}

def main [program: path, --summary: path = $default_summary] {
  let absent = (absent-names (open --raw $program) (open --raw $summary))
  if ($absent | is-empty) {
    print "primitive-coverage: 0 absent"
  } else {
    $absent | each {|n| print $n }
    exit 1
  }
}
