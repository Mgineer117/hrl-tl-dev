from __future__ import annotations

import sys

from absl import flags

if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)
