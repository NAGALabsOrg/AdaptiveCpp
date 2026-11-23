import os
import shutil

import lit.formats

config.name = "sg"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".cpp"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(os.environ.get("TMPDIR", "/tmp"), "sg-lit")

for tool in ("FileCheck", "not"):
    path = shutil.which(tool)
    if path:
        config.substitutions.append((tool, path))

config.substitutions.append(("%acpp", "acpp --acpp-targets=generic -std=c++20 -c -o %t.o"))

for var in ("ACPP_VISIBILITY_MASK", "HOME", "CONDA_PREFIX", "PATH"):
    if var in os.environ:
        config.environment[var] = os.environ[var]
