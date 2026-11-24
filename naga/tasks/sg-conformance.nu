# Check the repository against naga/work-manifest.md.
#
#   nu naga/tasks/sg-conformance.nu [--repo <dir>] [--base <commit>] [--manifest <path>]
#
# Findings, one per line, exit code 1 when there are any:
#   unlisted-file       a file added since the base that is in no row
#   duplicate-file      a file in more than one row
#   missing-interface   an interface absent from its header with no deviation entry naming it
#   bad-subject         a commit whose subject does not start with a unit id
#   foreign-file        a commit touching a file outside its unit's row

const default_base = "0dcb0a47"

export def parse-rows [manifest: string] {
  $manifest | lines | where {|l| $l =~ '^\| C\d+ \|' } | each {|l|
    let cells = ($l | split row '|' | each {|c| $c | str trim })
    {
      unit: ($cells | get 1)
      files: ($cells | get 2 | split row ',' | each {|f| $f | str trim } | where {|f| $f != "" })
      interfaces: (
        if ($cells | get 3) == "none" { [] } else {
          $cells | get 3 | split row ',' | each {|i| $i | str trim } | where {|i| $i != "" }
            | each {|i| let p = ($i | split row '@'); {symbol: ($p | get 0), header: ($p | get 1)} }
        }
      )
    }
  }
}

export def deviation-lines [manifest: string] {
  let ls = ($manifest | lines)
  let start = ($ls | enumerate | where {|e| $e.item starts-with "## Deviations" } | first | get index)
  $ls | skip ($start + 1) | where {|l| $l starts-with "- " }
}

# added: files added since the base. commits: table of {sha, subject, files}.
# header_text: closure from a repository-relative path to the text of that file ("" when absent).
export def findings [manifest: string, added: list<string>, commits: table, header_text: closure] {
  let rows = (parse-rows $manifest)
  let all_files = ($rows | each {|r| $r.files } | flatten)
  let unlisted = ($added | where {|f| $f not-in $all_files } | each {|f| {kind: "unlisted-file", name: $f} })
  let dup = ($all_files | uniq --repeated | each {|f| {kind: "duplicate-file", name: $f} })
  let devs = (deviation-lines $manifest)
  let missing = ($rows | each {|r|
    $r.interfaces | where {|i|
      let text = (do $header_text $i.header)
      let present = ($text =~ ('(?:^|[^A-Za-z0-9_])' + $i.symbol + '(?:[^A-Za-z0-9_]|$)'))
      let excused = ($devs | any {|d| $d | str contains $i.symbol })
      (not $present) and (not $excused)
    } | each {|i| {kind: "missing-interface", name: $"($i.symbol)@($i.header)"} }
  } | flatten)
  let by_commit = ($commits | each {|c|
    let unit = ($c.subject | parse --regex '^(?<id>C\d+):' | get -o 0.id)
    if $unit == null {
      [{kind: "bad-subject", name: $"($c.sha) ($c.subject)"}]
    } else {
      let row = ($rows | where unit == $unit)
      let own = (if ($row | is-empty) { [] } else { $row | first | get files })
      $c.files | where {|f| $f not-in $own } | each {|f| {kind: "foreign-file", name: $"($c.sha) ($unit) ($f)"} }
    }
  } | flatten)
  $unlisted | append $dup | append $missing | append $by_commit
}

export def collect-commits [repo: path, base: string] {
  let raw = (git -C $repo log --reverse --name-only --format=%x01%h%x02%s $"($base)..HEAD" | str join "\n")
  $raw | split row (char --integer 1) | where {|b| ($b | str trim) != "" } | each {|b|
    let ls = ($b | lines)
    let head = ($ls | first | split row (char --integer 2))
    {sha: ($head | get 0), subject: ($head | get 1), files: ($ls | skip 1 | where {|f| $f != "" })}
  }
}

def main [--repo: path = ".", --base: string = $default_base, --manifest: path = "naga/work-manifest.md"] {
  let added = (git -C $repo diff --name-only --diff-filter=A $"($base)..HEAD" | lines)
  let commits = (collect-commits $repo $base)
  let reader = {|p| let f = ($repo | path join $p); if ($f | path exists) { open --raw $f } else { "" } }
  let f = (findings (open --raw ($repo | path join $manifest)) $added $commits $reader)
  if ($f | is-empty) {
    print "sg-conformance: 0 findings"
  } else {
    $f | each {|x| print $"($x.kind) ($x.name)" }
    exit 1
  }
}
