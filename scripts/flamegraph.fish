set REPO_ROOT (dirname (dirname (readlink -m (status --current-filename))))

set WORKDIR /tmp/(dbus-uuidgen)
mkdir $WORKDIR
echo "straight.fish: workdir is $WORKDIR"
mkdir $WORKDIR/logs
mkdir $WORKDIR/scores

set BIN $WORKDIR/a.out
g++ $REPO_ROOT/main.cpp -std=c++17 -Wall -Wextra -o $BIN

cd $REPO_ROOT/tools

set f 0000.txt
sudo perf record -F 200 --call-graph dwarf -g ./target/release/tester in/$f $BIN
sudo perf script | perl $REPO_ROOT/FlameGraph/stackcollapse-perf.pl | perl $REPO_ROOT/FlameGraph/flamegraph.pl > flamegraph.svg
