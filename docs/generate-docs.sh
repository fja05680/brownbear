#!/bin/bash
# from brownbear/docs

# remove old html
rm -rf html

# generate html (repo root must be on PYTHONPATH)
export PYTHONPATH="$(cd .. && pwd)"
pdoc --html brownbear

# landing page for GitHub Pages (serves from /docs on main)
cat > index.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=html/brownbear/index.html">
  <title>brownbear documentation</title>
</head>
<body>
  <p><a href="html/brownbear/index.html">brownbear API documentation</a></p>
</body>
</html>
EOF

echo Done.
