#!/bin/bash
# Compare branches with a selected base (default: current checkout).
set -euo pipefail

usage() {
    printf 'Usage: %s [BRANCH | --base BRANCH]\n' "${0##*/}"
    printf 'Compare with BRANCH; default: current branch (HEAD when detached).\n'
}

base_arg=""
case $# in
    0) ;;
    1)
        case "$1" in
            -h|--help) usage; exit 0 ;;
            --base=*) base_arg=${1#--base=} ;;
            -*) usage >&2; exit 2 ;;
            *) base_arg=$1 ;;
        esac
        [[ -n "$base_arg" ]] || { usage >&2; exit 2; }
        ;;
    2)
        [[ "$1" == "--base" || "$1" == "-b" ]] || { usage >&2; exit 2; }
        base_arg=$2
        [[ -n "$base_arg" ]] || { usage >&2; exit 2; }
        ;;
    *) usage >&2; exit 2 ;;
esac

current_branch=$(git branch --show-current)
if [[ -n "$current_branch" ]]; then
    current_ref=$current_branch
else
    current_ref=HEAD
fi

# Freeze the resolved base so every row uses the same commit.
if [[ -z "$base_arg" || "$base_arg" == "HEAD" ]]; then
    base_ref=$current_ref
    comparison_commit=$(git rev-parse --verify HEAD)
elif git show-ref --verify --quiet "refs/heads/$base_arg"; then
    base_ref=$base_arg
    comparison_commit=$(git rev-parse --verify "refs/heads/${base_arg}^{commit}")
elif git show-ref --verify --quiet "refs/remotes/$base_arg"; then
    base_ref=$base_arg
    comparison_commit=$(git rev-parse --verify "refs/remotes/${base_arg}^{commit}")
else
    printf 'Unknown branch: %s\n' "$base_arg" >&2
    exit 2
fi


# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# 获取所有分支（本地+远程）
branches=$(git branch -a --format='%(refname:short)' | grep -v 'HEAD' | sort)
if [[ -z "$current_branch" && "$base_ref" != HEAD ]]; then
    branches=$(printf '%s\nHEAD\n' "$branches" | sort)
fi

# 获取 worktree 信息
worktree_info=$(git worktree list --porcelain)
current_wt=$(git rev-parse --show-toplevel 2>/dev/null)

# 函数：获取分支对应的 worktree 路径
get_worktree() {
    local branch=$1
    local path=""
    if [[ -z "$current_branch" && "$branch" == "HEAD" ]]; then
        echo "$current_wt"
        return
    fi
    while IFS= read -r line; do
        if [[ "$line" == "worktree "* ]]; then
            path="${line#worktree }"
        elif [[ "$line" == "branch "* ]]; then
            wt_branch="${line#branch }"
            if [[ "$wt_branch" == "refs/heads/$branch" ]]; then
                echo "$path"
                return
            fi
        fi
    done <<< "$worktree_info"
    echo ""
}

# 函数：简化 worktree 路径显示
simplify_wt() {
    local wt_path=$1
    if [[ -z "$wt_path" ]]; then
        echo "-"
    elif [[ "$wt_path" == "$current_wt" ]]; then
        echo "*current"
    else
        basename "$wt_path"
    fi
}

# 函数：获取分支最后一次提交时间
get_last_commit_time() {
    local branch=$1
    git log -1 --format='%ci' "$branch" | cut -d' ' -f1,2 | cut -d':' -f1,2
}

# 函数：获取分支最后一次提交的短 SHA
get_last_commit_id() {
    local branch=$1
    git rev-parse --short=12 "$branch"
}

# 计算最长分支名和最长 worktree 名
max_branch_len=0
max_wt_len=0

for branch in $branches; do
    len=${#branch}
    [[ $len -gt $max_branch_len ]] && max_branch_len=$len

    wt_path=$(get_worktree "$branch")
    wt_display=$(simplify_wt "$wt_path")
    wt_len=${#wt_display}
    wt_len=$((wt_len + 2))
    [[ $wt_len -gt $max_wt_len ]] && max_wt_len=$wt_len
done

[[ $max_branch_len -lt 20 ]] && max_branch_len=20
[[ $max_wt_len -lt 10 ]] && max_wt_len=10

max_branch_len=$((max_branch_len + 2))
max_wt_len=$((max_wt_len + 2))
ahead_len=7
behind_len=8
date_len=17
commit_id_len=12

# 计算分割线长度
total_len=$((max_branch_len + max_wt_len + ahead_len + behind_len + date_len * 2 + commit_id_len + 12))
separator=$(printf '%*s' "$total_len" '' | tr ' ' '-')

# 显示表头
printf 'Base: %s (%s)\n' "$base_ref" "${comparison_commit:0:12}"
printf "%-${max_branch_len}s  %-${max_wt_len}s  %-${ahead_len}s  %-${behind_len}s  %-${date_len}s  %-${date_len}s  %-${commit_id_len}s\n" \
    "BRANCH" "WORKTREE" "AHEAD" "BEHIND" "FORK" "COMMIT" "ID"
echo "$separator"

# 函数：显示一个分支
print_branch() {
    local branch=$1
    wt_path=$(get_worktree "$branch")
    wt_display=$(simplify_wt "$wt_path")
    last_commit_date=$(get_last_commit_time "$branch")
    last_commit_id=$(get_last_commit_id "$branch")

    if [[ "$branch" == "$base_ref" ]]; then
        ahead=0
        behind=0
        fork_date="-"
    else
        ahead=$(git rev-list --count "${comparison_commit}..${branch}" 2>/dev/null)
        behind=$(git rev-list --count "${branch}..${comparison_commit}" 2>/dev/null)
        if merge_base=$(git merge-base "$comparison_commit" "$branch"); then
            fork_date=$(git show -s --format='%ci' "$merge_base" | cut -d' ' -f1,2 | cut -d':' -f1,2)
        else
            status=$?
            [[ $status -eq 1 ]] || return "$status"
            fork_date="-"
        fi
    fi

    printf "%-${max_branch_len}s  %-${max_wt_len}s  " "$branch" "[$wt_display]"
    if [[ $ahead -gt 0 ]]; then
        printf "${GREEN}%-${ahead_len}d${NC}" "$ahead"
    else
        printf "%-${ahead_len}d" "$ahead"
    fi
    printf "  "
    if [[ $behind -gt 0 ]]; then
        printf "${RED}%-${behind_len}d${NC}" "$behind"
    else
        printf "%-${behind_len}d" "$behind"
    fi
    printf "  %-${date_len}s  %-${date_len}s  %-${commit_id_len}s\n" \
        "$fork_date" "$last_commit_date" "$last_commit_id"
}

# 显示基准分支（默认当前分支或 detached HEAD）
print_branch "$base_ref"
echo "$separator"

# 显示本地分支
for branch in $branches; do
    [[ "$branch" == "$base_ref" ]] && continue
    [[ "$branch" == origin/* ]] && continue
    print_branch "$branch"
done

echo "$separator"

# 显示远程分支
for branch in $branches; do
    [[ "$branch" == "$base_ref" ]] && continue
    [[ "$branch" == origin/* ]] || continue
    print_branch "$branch"
done
