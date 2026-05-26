# Run this after creating a PRIVATE empty GitHub repository named source-aware-stabilized-soil-ml.
# Replace <YOUR_GITHUB_USERNAME> with your GitHub username.

$RepoUrl = "https://github.com/<YOUR_GITHUB_USERNAME>/source-aware-stabilized-soil-ml.git"

git init
git add .
git commit -m "Initial reproducibility package for stabilized soil ML study"
git branch -M main
git remote add origin $RepoUrl
git push -u origin main
