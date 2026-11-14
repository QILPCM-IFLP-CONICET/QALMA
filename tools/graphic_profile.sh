#!/bin/bash

BRANCH=`git rev-parse --abbrev-ref HEAD`
echo $BRANCH
PROFILE_MASTER=`echo prof/master/$1.prof`
PROFILE_BRANCH=`echo prof/$BRANCH/$1.prof`


echo "graphic profile from '$PROFILE_MASTER'"
echo "graphic profile from '$PROFILE_BRANCH'"
gprof2dot --color-nodes-by-selftime -f pstats "$PROFILE_MASTER"  |dot -Tsvg -o gprof_master.svg
gprof2dot --color-nodes-by-selftime -f pstats "$PROFILE_BRANCH"  |dot -Tsvg -o gprof_$BRANCH.svg
# gprof2dot -f pstats "$PROFILE_MASTER"  |dot -Tsvg -o gprof_master.svg
# gprof2dot -f pstats "$PROFILE_BRANCH"  |dot -Tsvg -o gprof_$BRANCH.svg


echo "graphic profiles in  gprof_master.svg and  gprof_$BRANCH.svg"
