# fetch-sources.nu: the source cache of the core build (a deviation from the
# acpp-toolchain staging build, which gets its sources from rattler-build's
# `source` section). A library on purpose: it defines NO `main`; build-stage.nu
# `source`s it.
#
# Each tree in pins.toml is downloaded, checked, extracted and patched ONCE into
# <NAGA_BUILD_DIR>/src/<name>, together with a key file recording what produced
# it (url, sha256, the content of every patch). The tree is rebuilt from scratch,
# never re-patched in place, only when that key changes.
#
# <NAGA_BUILD_DIR>/src is the SRC_DIR layout build-stage.nu expects:
#   llvm-project OpenCL-Headers OpenCL-CLHPP rocm-dist   fetched trees
#   AdaptiveCpp                                          symlink to the fork working tree
#   shared                                               symlink to packages/core (scripts, licenses)

def tree-key [pin: record, patches_dir: string] {
  let patch_part = ($pin.patches? | default [] | each {|p|
    $"($p):(open --raw ($patches_dir | path join $p) | hash sha256)"
  } | str join "\n")
  $"($pin.url)\n($pin.sha256? | default '')\n($patch_part)" | hash sha256
}

def fetch-tree [name: string, pin: record, patches_dir: string, cache: string] {
  let src = ($cache | path join "src")
  let keys = ($cache | path join "keys")
  let dest = ($src | path join $name)
  let key_file = ($keys | path join $name)
  let key = (tree-key $pin $patches_dir)

  if ($dest | path exists) and ($key_file | path exists) and ((open --raw $key_file | str trim) == $key) {
    print $"fetch-sources: ($name) up to date"
    return
  }

  print $"fetch-sources: ($name) <- ($pin.url)"
  mkdir $src $keys
  let tmp = ($cache | path join "tmp" $name)
  rm --recursive --force $tmp $dest $key_file
  mkdir $tmp

  let archive = ($cache | path join "downloads" ($pin.url | path basename))
  mkdir ($archive | path dirname)
  ^curl --fail --location --retry 3 --silent --show-error --output $archive $pin.url

  if ($pin.sha256? | is-not-empty) {
    let got = (^sha256sum $archive | split row " " | first)
    if $got != $pin.sha256 {
      error make {msg: $"fetch-sources: ($name): sha256 of ($archive) is ($got), pinned ($pin.sha256)"}
    }
  }

  # The archive's single top directory is stripped, as rattler-build does for a url source.
  ^tar -xzf $archive -C $tmp --strip-components=1
  rm --force $archive

  for p in ($pin.patches? | default []) {
    print $"fetch-sources: ($name): patch ($p)"
    ^patch -p1 --no-backup-if-mismatch -d $tmp -i ($patches_dir | path join $p)
  }

  mv $tmp $dest
  $key | save --force $key_file
}

# Fetches what changed and returns the SRC_DIR layout directory.
def prepare-sources [] {
  let source_dir = $env.NAGA_SOURCE_DIR
  let core = ($source_dir | path join "packages" "core")
  let cache = $env.NAGA_BUILD_DIR
  let pins = (open ($core | path join "pins.toml"))
  let patches_dir = ($core | path join "patches")

  for name in ($pins | columns) {
    fetch-tree $name ($pins | get $name) $patches_dir $cache
  }

  let src = ($cache | path join "src")
  ^ln -sfn $source_dir ($src | path join "AdaptiveCpp")
  ^ln -sfn $core ($src | path join "shared")
  $src
}
