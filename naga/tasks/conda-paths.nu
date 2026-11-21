# Print info/paths.json of a .conda archive as JSON.
# Run: pixi exec -s unzip -s zstd -s tar -s nushell nu naga/tasks/conda-paths.nu <pkg.conda>
def main [pkg: path] {
  ^unzip -p $pkg "info-*.tar.zst" | ^zstd -d -q -c | ^tar -xO "info/paths.json"
}
