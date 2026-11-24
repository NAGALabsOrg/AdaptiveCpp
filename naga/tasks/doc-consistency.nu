# Compare the three design documents with the language summary.
#
#   nu naga/tasks/doc-consistency.nu [--summary <summary.md>] [--docs <dir>]
#
# Findings, one per line, exit code 1 when there are any:
#   missing-name     a primitive of the summary's table absent from 01, or a name of its graph section absent from 02
#   superseded-term  a superseded term present in one of 01-03
#   missing-id       an id of W1-W15, B1-B2, D2-D17 absent from 01-03 together

use primitive-coverage.nu [table-names, graph-names]

const default_summary = "docs/naga-internal/brainstorm/language-pass/summary.md"
const default_docs = "docs/naga-internal"
const superseded = [parallel_for map_groups group_expr mapping durable_graph promise future "structural hash" "key tuple"]

export def required-ids [] {
  [...(1..15 | each {|i| $"W($i)" }) B1 B2 ...(2..17 | each {|i| $"D($i)" })]
}

def has-name [text: string, name: string] {
  let literal = ($name | str replace --all '.' '[.]' | str replace --all '<' '[<]' | str replace --all '>' '[>]')
  $text =~ ($literal + '(?:[^A-Za-z0-9_]|$)')
}

def has-id [text: string, id: string] {
  $text =~ ('(?:^|[^A-Za-z0-9_])' + $id + '(?:[^A-Za-z0-9_]|$)')
}

export def findings [summary: string, d01: string, d02: string, d03: string] {
  let missing_primitives = (table-names $summary | where {|n| not (has-name $d01 $n) }
    | each {|n| {kind: "missing-name", where: "01", name: $n} })
  let missing_graph = (graph-names $summary | where {|n| not (has-name $d02 $n) }
    | each {|n| {kind: "missing-name", where: "02", name: $n} })
  let docs = [[where text]; ["01" $d01] ["02" $d02] ["03" $d03]]
  let old = ($docs | each {|d|
    $superseded | where {|t| $d.text =~ ('(?i)' + $t) } | each {|t| {kind: "superseded-term", where: $d.where, name: $t} }
  } | flatten)
  let all = ($d01 + "\n" + $d02 + "\n" + $d03)
  let ids = (required-ids | where {|i| not (has-id $all $i) } | each {|i| {kind: "missing-id", where: "01-03", name: $i} })
  $missing_primitives | append $missing_graph | append $old | append $ids
}

def main [--summary: path = $default_summary, --docs: path = $default_docs] {
  let f = (findings (open --raw $summary)
    (open --raw ($docs | path join 01-kernel-language.md))
    (open --raw ($docs | path join 02-graph-api.md))
    (open --raw ($docs | path join 03-vocabulary.md)))
  if ($f | is-empty) {
    print "doc-consistency: 0 findings"
  } else {
    $f | each {|x| print $"($x.kind) ($x.where) ($x.name)" }
    exit 1
  }
}
