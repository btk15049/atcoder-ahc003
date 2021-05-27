set REPO_ROOT (dirname (dirname (readlink -m (status --current-filename))))


set WORKDIR /tmp/(dbus-uuidgen)
mkdir $WORKDIR
echo "straight.fish: workdir is $WORKDIR"
mkdir $WORKDIR/logs
mkdir $WORKDIR/scores

set BIN $WORKDIR/a.out
g++ $REPO_ROOT/main.cpp -std=c++17 -Wall -Wextra -o $BIN

cd $REPO_ROOT/tools

for f in (ls in/)
    # echo $f
    cargo run --release --bin tester in/$f $BIN 2>$WORKDIR/logs/$f.log | tee $WORKDIR/scores/$f
end

echo "score:" (python3 $REPO_ROOT/py-tools/score.py $WORKDIR/scores)
