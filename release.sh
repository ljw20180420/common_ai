#!/bin/bash

uv version --bump patch
version="v$(uv version | cut -d" " -f2)"
git commit -am "release ${version}"
git push
gh release create "${version}" \
    --generate-notes \
    --fail-on-no-commits
git fetch --tags origin
