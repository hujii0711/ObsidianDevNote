git fetch origin develop
git log --oneline HEAD..origin/develop | head -20
git merge origin/develop --no-edit
git log --oneline -5