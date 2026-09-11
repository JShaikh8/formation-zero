.PHONY: site assets preview test

# Compile docs/ into the static tracker site (stdlib only).
site:
	python3 site/build.py

# Regenerate gallery SVGs from real project data (needs the project venv).
assets:
	.venv/bin/python scripts/gen_gallery_assets.py

# Serve the site locally.
preview: site
	python3 -m http.server 8000 --directory site

test:
	.venv/bin/python -m pytest -q
