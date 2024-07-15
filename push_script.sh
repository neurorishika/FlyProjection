remote=origin
branch=main
step=1
step_commits=$(git rev-list --reverse ${branch} | awk "NR % ${step} == 0")

for commit in ${step_commits} ${branch}; do git push ${remote} ${commit}:${branch}; done
